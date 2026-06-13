/**
 * Vision-MCP v7: analyze_pdf tool
 * PDF analysis with concurrent, multi-image, and batch strategies
 */

import { statSync } from "fs";
import pLimit from "p-limit";
import type { VisionProvider } from "../providers/base.js";
import type { PageResult, PdfStrategy, ProcessingSummary, FieldSpec } from "../config/types.js";
import {
  CONCURRENCY, WEBP_QUALITY, WEBP_QUALITY_OCR, MAX_IMAGE_WIDTH, MIN_OCR_QUALITY, PREPROCESS_WITH_THINKING,
  PDF_CHUNK_SIZE, TOKEN_BUDGET_ENABLED, TOKEN_BUDGET_K,
  ENABLE_PREPROCESSING, ENABLE_LAYERED_EXTRACTION, ENABLE_THINKING
, MODEL, IMAGE_PDF_LOSSLESS } from "../config/constants.js";
import { renderPageSmart, getPdfPageCount, getPdfType } from "../rendering/pdf.js";
import { optimizeForVision } from "../rendering/image.js";
import { retryWithBackoff } from "../utils/retry.js";
import { parsePageRange, chunkArray, isDocExtractionPrompt, isHandwritingPrompt, isFieldExtractionPrompt } from "../utils/helpers.js";
import { packImagesByTokenBudget } from "../utils/tokens.js";
import { enhancePrompt } from "./helpers.js";
import { extractLosslessDocument } from "../extraction/lossless.js";

export async function handleAnalyzePdf(
  provider: VisionProvider,
  args: any
): Promise<{ content: { type: "text"; text: string }[] }> {
  const path = args.pdf_path as string;
  const pagesRaw = (args.pages as string) || "";
  const prompt = args.prompt as string;
  const mt = (args.max_tokens as number) ?? 4096;
  const concurrency = (args.concurrency as number) ?? CONCURRENCY;
  const imgQuality = (args.image_quality as number) || (isDocExtractionPrompt(prompt) ? WEBP_QUALITY_OCR : WEBP_QUALITY);
  const imgMaxWidth = (args.max_image_width as number) ?? MAX_IMAGE_WIDTH;
  const enableThinking = args.enable_thinking as boolean | undefined;
  const thinkingBudget = args.thinking_budget as number | undefined;
  const vlHighRes = args.vl_high_resolution_images as boolean | undefined;
  const maxPixels = args.max_pixels as number | undefined;
  const minPixels = args.min_pixels as number | undefined;
  const forceStrategy = args.strategy as string | undefined;
  const chunkSizeOverride = (args.chunk_size as number) || PDF_CHUNK_SIZE;
  const rawFields = args.fields as any[] | undefined;
  const scVotes = (args.self_consistency_votes as number) || 3;
  const temperature = args.temperature as number | undefined;
  const topP = args.top_p as number | undefined;
  const extractionOptions = {
    cost_policy: args.cost_policy,
    cache_policy: args.cache_policy,
    return_cost_breakdown: args.return_cost_breakdown,
    max_unverified_required_fields: args.max_unverified_required_fields,
    ocr_verify: args.ocr_verify,
  };

  if (!path) return { content: [{ type: "text", text: "Error: pdf_path required" }] };
  if (!prompt) return { content: [{ type: "text", text: "Error: prompt required" }] };

  const total = await getPdfPageCount(path);
  const nums = pagesRaw ? parsePageRange(pagesRaw, total) : Array.from({ length: total }, (_, i) => i + 1);
  if (!nums.length) return { content: [{ type: "text", text: `Error: No valid pages in "${pagesRaw}" (total: ${total})` }] };

  const strategy = selectStrategy(nums.length, forceStrategy);
  console.error(`[analyze_pdf] ${nums.length} pages, strategy=${strategy}`);

  const enhancedPrompt = enhancePrompt(prompt);

  switch (strategy) {
    case "batch":
      return handleBatchStrategy(provider, path, nums, enhancedPrompt, mt, imgQuality, imgMaxWidth, enableThinking, thinkingBudget);

    case "multi-image":
      return handleMultiImageStrategy(provider, path, nums, enhancedPrompt, mt, concurrency, imgQuality, imgMaxWidth, enableThinking, thinkingBudget, vlHighRes, maxPixels, minPixels, temperature, topP, chunkSizeOverride, rawFields, scVotes, extractionOptions);

    case "concurrent":
      return handleConcurrentStrategy(provider, path, nums, enhancedPrompt, mt, concurrency, imgQuality, imgMaxWidth, enableThinking, thinkingBudget, vlHighRes, maxPixels, minPixels, temperature, topP, rawFields, scVotes, extractionOptions);

    default:
      return { content: [{ type: "text", text: `Error: Unknown strategy: ${strategy}` }] };
  }
}

function selectStrategy(pageCount: number, force?: string): PdfStrategy {
  if (force === "concurrent") return "concurrent";
  if (force === "multi-image") return "multi-image";
  if (force === "batch") return "batch";
  if (pageCount <= 50) return "concurrent";
  if (pageCount <= 200) return "multi-image";
  return "batch";
}

async function handleBatchStrategy(
  provider: VisionProvider,
  path: string,
  nums: number[],
  prompt: string,
  mt: number,
  imgQuality: number,
  imgMaxWidth: number,
  enableThinking?: boolean,
  thinkingBudget?: number
) {
  const { submitBatchedJobs } = await import("../batch/manager.js");
  const { BATCH_MAX_PAGES } = await import("../config/constants.js");
  const chunks = chunkArray(nums, BATCH_MAX_PAGES);
  console.error(`[analyze_pdf] Splitting ${nums.length} pages into ${chunks.length} batch jobs`);

  const { batchIds } = await submitBatchedJobs(
    provider, path, nums, prompt, mt, imgQuality, imgMaxWidth, enableThinking, thinkingBudget
  );

  return {
    content: [{
      type: "text" as const,
      text: JSON.stringify({
        strategy: "batch",
        total_pages: nums.length,
        chunks: chunks.length,
        batch_ids: batchIds,
        message: "Batch jobs submitted. Poll with get_batch_status or get_batch_status_all."
      }, null, 2)
    }]
  };
}

async function handleConcurrentStrategy(
  provider: VisionProvider,
  path: string,
  nums: number[],
  prompt: string,
  mt: number,
  concurrency: number,
  imgQualityIn: number,
  imgMaxWidth: number,
  enableThinking?: boolean,
  thinkingBudget?: number,
  vlHighRes?: boolean,
  maxPixels?: number,
  minPixels?: number,
  temperature?: number,
  topP?: number,
  rawFields?: any[],
  scVotes?: number,
  extractionOptions: Record<string, any> = {}
) {
  if (nums.length > 200) {
    return { content: [{ type: "text" as const, text: `Error: Too many pages (${nums.length}) for concurrent strategy. Use strategy=multi-image or strategy=batch.` }] };
  }

  const limit = pLimit(concurrency);
  const tStart = Date.now();
  console.error(`[analyze_pdf] Processing ${nums.length} pages (concurrency=${concurrency})...`);

  const enhancedPrompt = enhancePrompt(prompt);

  // v8.2: quality guard
  const _isDocC = isDocExtractionPrompt(prompt);
  const imgQuality = (_isDocC && imgQualityIn < MIN_OCR_QUALITY) ? MIN_OCR_QUALITY : imgQualityIn;
  const qualityAdjustedC = _isDocC && imgQualityIn < MIN_OCR_QUALITY;
  
  // Render all pages first
  const prepared = await Promise.all(
    nums.map((pn) =>
      limit(async () => {
        try {
          const img = await renderPageSmart(path, pn, imgMaxWidth, ENABLE_PREPROCESSING, isHandwritingPrompt(prompt), !PREPROCESS_WITH_THINKING && (enableThinking ?? false));
          const isIb = IMAGE_PDF_LOSSLESS ? (getPdfType(path) === "image") : false; const opt = await optimizeForVision(img, { quality: imgQuality, maxWidth: imgMaxWidth, isImageBasedPdf: isIb });
          return { page: pn, buffer: opt.buffer, mime: opt.mime, error: null as string | null };
        } catch (err: any) {
          return { page: pn, buffer: null as any, mime: "", error: err.message };
        }
      })
    )
  );
  prepared.sort((a, b) => a.page - b.page);

  const visionTasks = prepared
    .filter((x) => !x.error)
    .map((x) =>
      limit(async () => {
        try {
          const res = rawFields
            ? await retryWithBackoff(
                () => extractLosslessDocument(provider, x.buffer, x.mime, rawFields, {
                  page: x.page,
                  sourcePath: path,
                  maxTokens: mt,
                  vlHighResolutionImages: vlHighRes,
                  returnCostBreakdown: extractionOptions.return_cost_breakdown !== false,
                  maxUnverifiedRequiredFields: extractionOptions.max_unverified_required_fields,
                  costPolicy: extractionOptions.cost_policy,
                  cachePolicy: extractionOptions.cache_policy,
                }),
                `Page ${x.page} (lossless)`
              )
            : await retryWithBackoff(
                () =>
                  provider.chat({
                    model: MODEL,
                    max_tokens: mt,
                    messages: [
                      {
                        role: "user",
                        content: [
                          { type: "image_url", image_url: { url: `data:${x.mime};base64,${x.buffer.toString("base64")}` } },
                          { type: "text", text: `[Page ${x.page}]\n${enhancedPrompt}` },
                        ],
                      },
                    ],
                    enable_thinking: enableThinking,
                    thinking_budget: thinkingBudget,
                    vl_high_resolution_images: vlHighRes,
                    max_pixels: maxPixels,
                    min_pixels: minPixels,
                    temperature,
                    top_p: topP,
                  }),
                `Page ${x.page}`
              );

          if (rawFields !== undefined) {
            const lr = res as any;
            return { page: x.page, success: true, text: JSON.stringify(lr), reasoning: undefined, input_tokens: lr.costBreakdown?.[0]?.input_tokens, output_tokens: lr.costBreakdown?.[0]?.output_tokens } as PageResult;
          }
          const vr = res as any;
          return { page: x.page, success: true, text: vr.text, reasoning: vr.reasoning, input_tokens: vr.it, output_tokens: vr.ot } as PageResult;
        } catch (err: any) {
          return { page: x.page, success: false, text: "", error: err.message } as PageResult;
        }
      })
    );

  const results = await Promise.all(visionTasks);
  for (const x of prepared) {
    if (x.error) results.push({ page: x.page, success: false, text: "", error: x.error });
  }
  results.sort((a, b) => a.page - b.page);

  const ok = results.filter((r) => r.success).length;
  const ti = results.reduce((s, r) => s + (r.input_tokens || 0), 0);
  const to = results.reduce((s, r) => s + (r.output_tokens || 0), 0);
  const elapsed = ((Date.now() - tStart) / 1000).toFixed(1);
  const errs = results.filter((r) => !r.success).map((r) => `Page ${r.page}: ${r.error}`);
  console.error(`[analyze_pdf] Done: ${ok}/${nums.length} in ${elapsed}s | tokens: ${ti}+${to}`);

  const summary: ProcessingSummary = {
    strategy: "concurrent",
    pipeline: {
      preprocessing: ENABLE_PREPROCESSING,
      layered_extraction: rawFields !== undefined,
      structured_output: rawFields !== undefined,
      lossless_extraction: rawFields !== undefined,
    },
    requested: nums.length,
    successful: ok,
    failed: results.length - ok,
    total_input_tokens: ti,
    total_output_tokens: to,
    elapsed_seconds: parseFloat(elapsed),
    concurrency,
  };

  return {
    content: [{
      type: "text" as const,
      text: JSON.stringify({
        ...(qualityAdjustedC ? { quality_adjusted: true, adjusted_from: imgQualityIn } : {}),
        summary, results, ...(errs.length ? { errors: errs } : {})
      }, null, 2)
    }]
  };
}

async function handleMultiImageStrategy(
  provider: VisionProvider,
  path: string,
  nums: number[],
  prompt: string,
  mt: number,
  concurrency: number,
  imgQualityIn: number,
  imgMaxWidth: number,
  enableThinking?: boolean,
  thinkingBudget?: number,
  vlHighRes?: boolean,
  maxPixels?: number,
  minPixels?: number,
  temperature?: number,
  topP?: number,
  chunkSizeOverride?: number,
  rawFields?: any[],
  scVotes?: number,
  extractionOptions: Record<string, any> = {}
) {
  const limit = pLimit(concurrency);
  const tStart = Date.now();

  // v8.2: quality guard
  const _isDocM = isDocExtractionPrompt(prompt);
  const imgQuality = (_isDocM && imgQualityIn < MIN_OCR_QUALITY) ? MIN_OCR_QUALITY : imgQualityIn;
  const qualityAdjustedM = _isDocM && imgQualityIn < MIN_OCR_QUALITY;
  
  // Render all pages concurrently
  console.error(`[analyze_pdf] Multi-image: rendering ${nums.length} pages...`);
  const preparedAll = await Promise.all(
    nums.map((pn) =>
      limit(async () => {
        try {
          const img = await renderPageSmart(path, pn, imgMaxWidth, ENABLE_PREPROCESSING, isHandwritingPrompt(prompt), !PREPROCESS_WITH_THINKING && (enableThinking ?? false));
          const isIb = IMAGE_PDF_LOSSLESS ? (getPdfType(path) === "image") : false; const opt = await optimizeForVision(img, { quality: imgQuality, maxWidth: imgMaxWidth, isImageBasedPdf: isIb });
          return { page: pn, buffer: opt.buffer, mime: opt.mime, width: img.width, height: img.height, error: null as string | null };
        } catch (err: any) {
          return { page: pn, buffer: null as any, mime: "", width: 0, height: 0, error: err.message };
        }
      })
    )
  );
  preparedAll.sort((a, b) => a.page - b.page);

  const renderOk = preparedAll.filter((x) => !x.error);
  const renderFailed = preparedAll.filter((x) => x.error);
  console.error(`[analyze_pdf] Rendered: ${renderOk.length} ok, ${renderFailed.length} failed`);

  // Pack images by token budget or fixed chunk
  let packs = TOKEN_BUDGET_ENABLED
    ? packImagesByTokenBudget(renderOk, TOKEN_BUDGET_K)
    : chunkArray(renderOk, chunkSizeOverride || PDF_CHUNK_SIZE);

  // Process each pack
  const allResults = await Promise.all(
    packs.map((pack, ci) =>
      limit(async () => {
        const pageRange = pack.length === 1 ? `Page ${pack[0].page}` : `Pages ${pack[0].page}-${pack[pack.length - 1].page}`;
        console.error(`[analyze_pdf] Pack ${ci + 1}/${packs.length}: ${pageRange} (${pack.length} images)`);
        try {
          if (rawFields !== undefined) {
            const pageResults = await Promise.all(pack.map((p) =>
              retryWithBackoff(
                () => extractLosslessDocument(provider, p.buffer, p.mime, rawFields, {
                  page: p.page,
                  sourcePath: path,
                  maxTokens: mt,
                  vlHighResolutionImages: vlHighRes,
                  returnCostBreakdown: extractionOptions.return_cost_breakdown !== false,
                  maxUnverifiedRequiredFields: extractionOptions.max_unverified_required_fields,
                  costPolicy: extractionOptions.cost_policy,
                  cachePolicy: extractionOptions.cache_policy,
                }),
                `Page ${p.page} (lossless)`
              ).then((res: any) => ({
                page: p.page,
                success: true,
                text: JSON.stringify(res),
                input_tokens: res.costBreakdown?.[0]?.input_tokens,
                output_tokens: res.costBreakdown?.[0]?.output_tokens,
              } as PageResult))
            ));
            return pageResults;
          }
          const images = pack.map((p) => ({ buf: p.buffer, mime: p.mime }));
          const pageList = pack.map((p) => p.page).join(",");
          const res = await retryWithBackoff(
            () =>
              provider.chatMultiImage(images, `[Pages ${pageList}]\n${prompt}`, {
                model: MODEL,
                max_tokens: mt,
                enable_thinking: enableThinking,
                thinking_budget: thinkingBudget,
                vl_high_resolution_images: vlHighRes,
                max_pixels: maxPixels,
                min_pixels: minPixels,
                temperature,
                top_p: topP,
              }),
            `Pack ${ci + 1}`
          );
          return pack.map((p) => ({
            page: p.page,
            success: true,
            text: res.text,
            reasoning: res.reasoning,
            input_tokens: res.it ? Math.round(res.it / pack.length) : undefined,
            output_tokens: res.ot ? Math.round(res.ot / pack.length) : undefined,
          } as PageResult));
        } catch (err: any) {
          return pack.map((p) => ({ page: p.page, success: false, text: "", error: err.message } as PageResult));
        }
      })
    )
  );

  // Merge render failures
  for (const rf of renderFailed) {
    allResults.push([{ page: rf.page, success: false, text: "", error: rf.error! }]);
  }
  const flatResults = allResults.flat().sort((a, b) => a.page - b.page);
  const ok = flatResults.filter((r) => r.success).length;
  const elapsed = ((Date.now() - tStart) / 1000).toFixed(1);
  const errs = flatResults.filter((r) => !r.success).map((r) => `Page ${r.page}: ${r.error}`);
  console.error(`[analyze_pdf] Multi-image done: ${ok}/${nums.length} in ${elapsed}s`);

  const budgetLabel = TOKEN_BUDGET_ENABLED ? `multi-image (token-budget ${TOKEN_BUDGET_K}K)` : "multi-image";
  return {
    content: [{
      type: "text" as const,
      text: JSON.stringify({
        ...(qualityAdjustedM ? { quality_adjusted: true, adjusted_from: imgQualityIn } : {}),
        summary: {
          strategy: budgetLabel,
          pipeline: { preprocessing: ENABLE_PREPROCESSING, layered_extraction: rawFields !== undefined, structured_output: rawFields !== undefined, lossless_extraction: rawFields !== undefined },
          requested: nums.length,
          successful: ok,
          failed: nums.length - ok,
          elapsed_seconds: parseFloat(elapsed),
        },
        results: flatResults,
        ...(errs.length ? { errors: errs } : {})
      }, null, 2)
    }]
  };
}


