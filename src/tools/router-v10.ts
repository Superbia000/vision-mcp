import { existsSync, readFileSync, statSync, writeFileSync } from "fs";
import { dirname, extname, join, parse } from "path";
import sharp from "sharp";
import type { VisionProvider } from "../providers/base.js";
import {
  BATCH_MAX_PAGES,
  MAX_IMAGE_WIDTH,
  MAX_OUTPUT_TOKENS,
  MODEL,
  WEBP_QUALITY_OCR,
} from "../config/constants.js";
import { getCapabilityMatrix } from "../runtime/capabilities.js";
import { resolveAccuracyPolicy } from "../runtime/accuracy.js";
import {
  asTextEnvelope,
  collectTokenMetrics,
  envelope,
  parseJsonText,
  timed,
} from "../runtime/envelope.js";
import { handleEstimateTokens, handlePdfInfo, handleBatchStatus, handleBatchStatusAll } from "./common.js";
import { handleAnalyzeImage } from "./image-analyze.js";
import { handleAnalyzePdf } from "./pdf-analyze.js";
import { handleAnalyzeVideo, handleAnalyzeVideoChunked } from "./video.js";
import { handleExtractFields, handleExtractVerify, handleHandwriting } from "./extraction.js";
import { aggregateLosslessPages } from "../extraction/lossless.js";
import { submitBatchedJobs } from "../batch/manager.js";
import { getPdfPageCount, getPdfType, renderPageSmart } from "../rendering/pdf.js";
import {
  preprocessAggressive,
  preprocessForOCR,
  preprocessHandwriting,
  preprocessLight,
  preprocessNegative,
  preprocessScanned,
  preprocessTable,
} from "../preprocessing/pipeline.js";
import { extToMime, isImageExt, isVideoExt, parsePageRange } from "../utils/helpers.js";
import { optimizeForVision } from "../rendering/image.js";

type ToolResult = { content: { type: "text"; text: string }[] };

export class ToolRouter {
  constructor(private readonly provider: VisionProvider) {}

  async dispatch(name: string, args: any): Promise<ToolResult> {
    switch (name) {
      case "vision_inspect":
        return this.visionInspect(args);
      case "vision_prepare":
        return this.visionPrepare(args);
      case "vision_analyze":
        return this.visionAnalyze(args);
      case "vision_extract":
        return this.visionExtract(args);
      case "vision_jobs":
        return this.visionJobs(args);
      default:
        return asTextEnvelope(envelope({
          success: false,
          tool: name,
          strategy: "unknown",
          errors: [`Unknown tool: ${name}`],
        }));
    }
  }

  private ensureFile(filePath: string): string | null {
    if (!filePath) return "file_path required";
    if (!existsSync(filePath)) return `File not found: ${filePath}`;
    return null;
  }

  private mediaType(filePath: string, override?: string): "pdf" | "image" | "video" | "unknown" {
    if (override && override !== "auto") return override as any;
    const ext = extname(filePath).toLowerCase();
    if (ext === ".pdf") return "pdf";
    if (isImageExt(ext)) return "image";
    if (isVideoExt(ext)) return "video";
    return "unknown";
  }

  private translatePdfStrategy(strategy?: string): string | undefined {
    if (!strategy || strategy === "auto") return undefined;
    if (strategy === "realtime") return "concurrent";
    if (strategy === "multi_image" || strategy === "chunked") return "multi-image";
    if (strategy === "batch") return "batch";
    return strategy;
  }

  private parseToolJson(result: string | ToolResult): any {
    const text = typeof result === "string" ? result : result.content?.[0]?.text ?? "";
    return parseJsonText(text);
  }

  private summarizeErrors(result: any): string[] | undefined {
    if (!result) return undefined;
    if (typeof result.error === "string") return [result.error];
    if (Array.isArray(result.errors)) return result.errors.map((e: any) => String(e));
    return undefined;
  }

  private markReviewFields(result: any): any {
    const clone = result && typeof result === "object" ? structuredClone(result) : result;
    const visit = (node: any) => {
      if (!node || typeof node !== "object") return;
      if ("confidence" in node || "verified" in node || "value" in node) {
        const confidence = String(node.confidence ?? "").toLowerCase();
        const verified = node.verified === true;
        const value = String(node.value ?? "");
        if (!verified || confidence === "low" || value.includes("[?]") || value.trim() === "") {
          node.needs_review = true;
        } else {
          node.needs_review = false;
        }
      }
      for (const item of Object.values(node)) visit(item);
    };
    visit(clone);
    return clone;
  }

  private async visionInspect(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({ success: false, tool: "vision_inspect", strategy: "validate", errors: [missing] }));
    }

    const policy = resolveAccuracyPolicy(args);
    const mediaType = this.mediaType(filePath);
    const capability = getCapabilityMatrix(this.provider);

    const { value, elapsedMs } = await timed(async () => {
      const estimates: Record<string, any> = {};
      if (mediaType === "pdf") {
        estimates.pdf_info = this.parseToolJson(await handlePdfInfo({ pdf_path: filePath }));
        estimates.tokens = this.parseToolJson(await handleEstimateTokens({
          file_path: filePath,
          pages: args.pages || "1",
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
        }));
        estimates.pdf_type = getPdfType(filePath) || "unknown";
      } else {
        estimates.tokens = this.parseToolJson(await handleEstimateTokens({
          file_path: filePath,
          pages: args.pages || "1",
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
          fps: args.fps,
        }));
        if (mediaType === "image") {
          const meta = await sharp(readFileSync(filePath)).metadata();
          estimates.image = {
            width: meta.width,
            height: meta.height,
            format: meta.format,
            size_mb: +(statSync(filePath).size / 1024 / 1024).toFixed(2),
          };
        }
      }
      return estimates;
    });

    const recommendedStrategy =
      mediaType === "pdf"
        ? "vision_analyze strategy=auto or vision_extract for fields"
        : mediaType === "video"
          ? "vision_analyze strategy=auto; use chunked for large videos"
          : "vision_analyze for free-form or vision_extract for fields";

    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_inspect",
      strategy: "inspect",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        recommended_strategy: recommendedStrategy,
        cost_controls: {
          image_token_policy: "Use full-page calls for layout/context, then crop OCR only for unverified or invalid fields.",
          cache_policy: args.cache_policy || "auto",
          cost_policy: args.cost_policy || "quality_first",
          quality_guard: "Cost hints never suppress required-field verification.",
          batch_note: "Batch can reduce cost for latency-insensitive jobs only when provider/region/model support is known.",
        },
        capabilities: capability,
      },
      results: value,
      metrics: { elapsed_ms: elapsedMs, ...collectTokenMetrics(value) },
      errors: this.summarizeErrors(value),
    }));
  }

  private async visionPrepare(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({ success: false, tool: "vision_prepare", strategy: "validate", errors: [missing] }));
    }

    const mode = (args.mode as string) || "auto";
    const mediaType = this.mediaType(filePath);
    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const total = await getPdfPageCount(filePath);
        const nums = args.pages ? parsePageRange(args.pages, total) : [1];
        const rendered = [];
        for (const pn of nums) {
          const img = await renderPageSmart(filePath, pn, MAX_IMAGE_WIDTH, true, mode === "handwriting");
          const outPath = this.resolveOutputPath(args.output_path, filePath, pn, "png");
          if (outPath) writeFileSync(outPath, img.buffer);
          rendered.push({
            page: pn,
            width: img.width,
            height: img.height,
            mime: img.mime,
            output_path: outPath,
            image_base64: args.return_base64 ? img.buffer.toString("base64") : undefined,
          });
        }
        return { media_type: "pdf", rendered };
      }

      if (mediaType !== "image") return { error: `Unsupported prepare file type: ${extname(filePath)}` };

      const raw = readFileSync(filePath);
      const prepared =
        mode === "light" ? await preprocessLight(raw)
          : mode === "table" ? await preprocessTable(raw)
            : mode === "handwriting" ? await preprocessHandwriting(raw)
              : mode === "negative" ? await preprocessNegative(raw)
                : mode === "scan" || mode === "photo" ? await preprocessForOCR(raw, { docType: mode as any })
                  : await preprocessForOCR(raw);
      const outPath = args.output_path as string | undefined;
      if (outPath) writeFileSync(outPath, Buffer.from(prepared.buffer));
      return {
        media_type: "image",
        mode,
        width: prepared.width,
        height: prepared.height,
        mime: prepared.mime,
        steps: prepared.appliedSteps,
        detected_type: prepared.detectedDocType,
        output_path: outPath,
        image_base64: args.return_base64 || !outPath ? Buffer.from(prepared.buffer).toString("base64") : undefined,
      };
    });

    return asTextEnvelope(envelope({
      success: !value?.error,
      tool: "vision_prepare",
      strategy: mode,
      summary: { media_type: mediaType },
      results: value,
      metrics: { elapsed_ms: elapsedMs },
      errors: value?.error ? [value.error] : undefined,
    }));
  }

  private resolveOutputPath(outputPath: string | undefined, inputPath: string, page: number, ext: string): string | undefined {
    if (!outputPath) return undefined;
    const parsed = parse(outputPath);
    if (parsed.ext && page === 1) return outputPath;
    if (parsed.ext) return join(parsed.dir || dirname(inputPath), `${parsed.name}-p${page}${parsed.ext}`);
    const base = parse(inputPath).name;
    return join(outputPath, `${base}-p${page}.${ext}`);
  }

  private async visionAnalyze(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const prompt = args.prompt as string;
    const missing = this.ensureFile(filePath);
    if (missing || !prompt) {
      return asTextEnvelope(envelope({
        success: false,
        tool: "vision_analyze",
        strategy: "validate",
        errors: [missing, !prompt ? "prompt required" : undefined].filter(Boolean) as string[],
      }));
    }

    const mediaType = this.mediaType(filePath, args.media_type);
    const policy = resolveAccuracyPolicy(args);
    const common = {
      prompt,
      pages: args.pages || "",
      max_tokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
      concurrency: policy.concurrency,
      chunk_size: args.chunk_size,
      image_quality: policy.imageQuality,
      max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
      enable_thinking: args.enable_thinking,
      thinking_budget: args.thinking_budget,
      vl_high_resolution_images: policy.vlHighResolutionImages,
      temperature: args.temperature,
      top_p: args.top_p,
    };

    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const strategy = this.translatePdfStrategy(args.strategy);
        const result = await handleAnalyzePdf(this.provider, { ...common, pdf_path: filePath, strategy });
        return this.parseToolJson(result);
      }
      if (mediaType === "image") {
        return this.parseToolJson(await handleAnalyzeImage(this.provider, { ...common, image_path: filePath }));
      }
      if (mediaType === "video") {
        const useChunked = args.strategy === "chunked";
        return this.parseToolJson(
          useChunked
            ? await handleAnalyzeVideoChunked(this.provider, { ...common, video_path: filePath, chunk_duration_sec: args.chunk_duration_sec, aggregate: args.aggregate })
            : await handleAnalyzeVideo(this.provider, { ...common, video_path: filePath, fps: args.fps, nframes: args.nframes })
        );
      }
      return { error: `Unsupported file type: ${extname(filePath)}` };
    });

    const metrics = collectTokenMetrics(value);
    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_analyze",
      strategy: args.strategy || value?.summary?.strategy || "auto",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        underlying_model: MODEL,
        ...(value?.summary ? { underlying_summary: value.summary } : {}),
      },
      results: value?.results ?? value?.text ?? value,
      metrics: { elapsed_ms: elapsedMs, ...metrics },
      warnings: getCapabilityMatrix(this.provider).notes,
      errors: this.summarizeErrors(value),
    }));
  }

  private async visionExtract(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const fields = args.fields as any[] | undefined;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({
        success: false,
        tool: "vision_extract",
        strategy: "validate",
        errors: [missing].filter(Boolean) as string[],
      }));
    }

    const mediaType = this.mediaType(filePath);
    const policy = resolveAccuracyPolicy(args);
    const prompt = this.buildFieldPrompt(fields || []);
    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const result = await handleAnalyzePdf(this.provider, {
          pdf_path: filePath,
          pages: args.pages || "1",
          prompt,
          fields: fields || [],
          lossless_extraction: true,
          strategy: args.strategy,
          max_tokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
          concurrency: policy.concurrency,
          image_quality: policy.imageQuality,
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
          vl_high_resolution_images: policy.vlHighResolutionImages,
          self_consistency_votes: policy.selfConsistencyVotes,
          temperature: args.temperature ?? 0,
          cost_policy: args.cost_policy,
          cache_policy: args.cache_policy,
          return_cost_breakdown: args.return_cost_breakdown !== false,
          budget_hint_usd: args.budget_hint_usd,
          max_unverified_required_fields: args.max_unverified_required_fields,
          ocr_verify: policy.ocrVerify,
        });
        const parsed = this.parseToolJson(result);
        if (parsed?.results) return aggregateLosslessPages(parsed.results, fields || [], filePath);
        return parsed;
      }
      if (mediaType === "image") {
        if (args.document_type === "handwriting" && fields?.length === 1 && /text|hand/i.test(fields[0].name || "")) {
          return this.parseToolJson(await handleHandwriting(this.provider, {
            image_path: filePath,
            prompt,
            language_hint: args.language_hint,
          }));
        }
        return this.parseToolJson(await handleExtractFields(this.provider, {
          image_path: filePath,
          fields,
          preserve_all: args.preserve_all !== false,
          output_schema: args.output_schema || "lossless_document_v1",
          max_tokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
          use_ocr_model: policy.ocrVerify,
          preprocess: args.preprocess !== false,
          strategy: args.strategy || (args.document_type && args.document_type !== "auto" ? undefined : "auto"),
          enable_thinking: args.enable_thinking,
          self_consistency_votes: policy.selfConsistencyVotes,
          cost_policy: args.cost_policy,
          cache_policy: args.cache_policy,
          return_cost_breakdown: args.return_cost_breakdown !== false,
          budget_hint_usd: args.budget_hint_usd,
          max_unverified_required_fields: args.max_unverified_required_fields,
        }));
      }
      return { error: `Unsupported extraction file type: ${extname(filePath)}` };
    });

    const reviewed = this.markReviewFields(value);
    const metrics = collectTokenMetrics(reviewed);
    const errors = this.summarizeErrors(reviewed);
    return asTextEnvelope(envelope({
      success: !errors && reviewed?.success !== false,
      tool: "vision_extract",
      strategy: mediaType === "pdf" ? "pdf-fields" : "document-fields",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        ocr_verify: policy.ocrVerify,
        return_evidence: args.return_evidence !== false,
        cost_policy: args.cost_policy || "quality_first",
        cache_policy: args.cache_policy || "auto",
        budget_hint_usd: args.budget_hint_usd,
      },
      results: reviewed,
      metrics: { elapsed_ms: elapsedMs, ...metrics },
      warnings: getCapabilityMatrix(this.provider).notes,
      errors,
    }));
  }

  private buildFieldPrompt(fields: any[]): string {
    const names = fields.map((f) => f.name || f.label_pattern || f.labelPattern).filter(Boolean).join(", ");
    return [
      "Perform lossless document extraction.",
      "Preserve every visible text item, table, field candidate, unknown field, and orphan value.",
      "After preserving all data, map requested fields when provided.",
      "Do not fabricate missing or obscured values. Use [?] for uncertain characters.",
      "Return JSON using schema lossless_document_v1.",
      `Requested fields: ${names || "(none; discover all visible fields)"}`,
    ].join("\n");
  }

  private async visionJobs(args: any): Promise<ToolResult> {
    const action = args.action as string;
    const capability = getCapabilityMatrix(this.provider);
    const { value, elapsedMs } = await timed(async () => {
      if (action === "status") {
        return this.parseToolJson(await handleBatchStatus(this.provider, { batch_id: args.job_id }));
      }
      if (action === "status_all") {
        return this.parseToolJson(await handleBatchStatusAll(this.provider, { batch_ids: args.job_ids }));
      }
      if (action === "results") {
        return this.parseToolJson(await handleBatchStatus(this.provider, { batch_id: args.job_id }));
      }
      if (action === "cancel") {
        return this.cancelBatch(args.job_id);
      }
      if (action === "list") {
        return this.listBatches();
      }
      if (action !== "submit") return { error: `Unsupported action: ${action}` };
      if (args.batch_policy === "disable") return { error: "batch_policy=disable; use vision_analyze realtime instead" };
      if (!capability.supportsBatch && args.batch_policy !== "force") {
        return {
          error: "Batch is not known to be supported for this provider/region/model.",
          fallback: "Use vision_analyze with strategy=realtime or strategy=multi_image.",
          capabilities: capability,
        };
      }
      if (!args.file_path || !args.prompt) return { error: "file_path and prompt required for submit" };
      if (this.mediaType(args.file_path) !== "pdf") return { error: "Only PDF batch submit is supported." };

      const total = await getPdfPageCount(args.file_path);
      const pages = args.pages ? parsePageRange(args.pages, total) : Array.from({ length: total }, (_, i) => i + 1);
      const policy = resolveAccuracyPolicy(args);
      const submitted = await submitBatchedJobs(
        this.provider,
        args.file_path,
        pages,
        args.prompt,
        args.max_tokens ?? MAX_OUTPUT_TOKENS,
        policy.imageQuality || WEBP_QUALITY_OCR,
        args.max_image_width ?? MAX_IMAGE_WIDTH,
        args.enable_thinking,
        args.thinking_budget,
      );
      return {
        total_pages: pages.length,
        batch_max_pages: BATCH_MAX_PAGES,
        batch_ids: submitted.batchIds,
        chunks: submitted.chunks,
        message: submitted.batchIds.length === 1
          ? "Batch submitted. Poll with vision_jobs action=status."
          : "Batches submitted. Poll with vision_jobs action=status_all.",
      };
    });

    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_jobs",
      strategy: action || "unknown",
      summary: { action, capabilities: capability },
      results: value,
      metrics: { elapsed_ms: elapsedMs, ...collectTokenMetrics(value) },
      errors: this.summarizeErrors(value),
    }));
  }

  private async cancelBatch(jobId: string | undefined): Promise<any> {
    if (!jobId) return { error: "job_id required" };
    const client = (this.provider as any).getClient?.();
    if (!client?.batches?.cancel) return { error: "Provider does not expose batches.cancel" };
    return client.batches.cancel(jobId);
  }

  private async listBatches(): Promise<any> {
    const client = (this.provider as any).getClient?.();
    if (!client?.batches?.list) return { error: "Provider does not expose batches.list" };
    return client.batches.list();
  }
}
