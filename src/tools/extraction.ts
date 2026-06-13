/**
 * Vision-MCP v8: Extraction tools
 *
 * v8: All extraction paths now route through layredExtract() for unified L1-L5.
 * Handwriting is now merged into layeredExtract with docType and languageHint.
 */

import { readFileSync, writeFileSync } from "fs";
import { extname } from "path";
import OpenAI from "openai";
import type { VisionProvider } from "../providers/base.js";
import type { FieldSpec, LayeredExtractionConfig, FieldValidationRule } from "../config/types.js";
import {
  MODEL, OCR_MODEL, BASE_URL, OCR_BASE_URL, API_KEY,
  ENABLE_THINKING, THINKING_BUDGET, FIX_MODEL,
  SELF_CONSISTENCY_VOTES, ENABLE_LAYERED_EXTRACTION,
  ENABLE_PREPROCESSING,
  MULTIPASS_ENABLED, MULTIPASS_VOTES,
} from "../config/constants.js";
import {
  preprocessForOCR, preprocessLight, preprocessAggressive,
  preprocessHandwriting, detectDocumentType,
} from "../preprocessing/pipeline.js";
import { layeredExtract } from "../extraction/layered.js";
import { extractLosslessDocument, toFieldSpecs } from "../extraction/lossless.js";
import { decideStrategy } from "../extraction/router.js";
import { extToMime, isImageExt } from "../utils/helpers.js";
import { enhancePrompt } from "./helpers.js";

// ---- ocr_enhance_image ----
export async function handleOcrEnhance(args: any): Promise<string> {
  const path = args.image_path as string;
  const mode = (args.mode as string) || "auto";
  const outputPath = args.output_path as string | undefined;

  if (!path) return JSON.stringify({ error: "image_path required" });

  try {
    const rawBuf = readFileSync(path);
    let result: any;
    if (mode === "light") {
      result = await preprocessLight(rawBuf);
    } else if (mode === "aggressive") {
      result = await preprocessAggressive(rawBuf);
    } else if (mode === "handwriting") {
      result = await preprocessHandwriting(rawBuf);
    } else {
      // auto: detect document type
      const detection = await detectDocumentType(rawBuf);
      result = await preprocessForOCR(rawBuf, {
        grayscale: true,
        removeBackground: true,
        enhanceContrast: true,
        sharpen: true,
        docType: detection.type,
      });
    }

    if (outputPath) {
      writeFileSync(outputPath, Buffer.from(result.buffer));
      return JSON.stringify({
        success: true, mode,
        detected_type: result.detectedDocType,
        output_path: outputPath,
        steps: result.appliedSteps,
        dimensions: `${result.width}x${result.height}`,
      }, null, 2);
    }

    return JSON.stringify({
      success: true, mode,
      detected_type: result.detectedDocType,
      steps: result.appliedSteps,
      dimensions: `${result.width}x${result.height}`,
      image_base64: Buffer.from(result.buffer).toString("base64"),
      mime: result.mime,
    }, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- extract_document_fields (v8: unified L1-L5) ----
export async function handleExtractFields(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const path = args.image_path as string;
  const rawFields = args.fields as any[] | undefined;
  const useOcrModel = args.use_ocr_model !== false;
  const enableThinking = args.enable_thinking as boolean | undefined;
  const scVotes = (args.self_consistency_votes as number) || SELF_CONSISTENCY_VOTES;
  const doPreprocess = args.preprocess !== false;
  const costPolicy = args.cost_policy as LayeredExtractionConfig["costPolicy"] | undefined;
  const cachePolicy = args.cache_policy as LayeredExtractionConfig["cachePolicy"] | undefined;

  if (!path) return JSON.stringify({ error: "image_path required" });

  try {
    let imageBuf: Buffer = readFileSync(path);
    let imageMime = extToMime(extname(path).toLowerCase());
    const preserveAll = args.preserve_all !== false;

    // v8.1: Detect document type BEFORE preprocessing for routing
    const docDetection = await detectDocumentType(imageBuf);
    const detectedDocType = docDetection.type;
    console.error(`[extract-fields] Detected document type: ${detectedDocType}`);
    const forcedStrategy = args.strategy as string | undefined;
    const routeDecision = decideStrategy(detectedDocType, forcedStrategy);

    if (preserveAll || !rawFields?.length) {
      const result = await extractLosslessDocument(provider, imageBuf, imageMime, rawFields, {
        sourcePath: path,
        docType: detectedDocType,
        preserveAll,
        maxTokens: args.max_tokens,
        returnCostBreakdown: args.return_cost_breakdown !== false,
        maxUnverifiedRequiredFields: typeof args.max_unverified_required_fields === "number" ? args.max_unverified_required_fields : undefined,
        costPolicy: args.cost_policy,
        cachePolicy: args.cache_policy,
      });
      return JSON.stringify(result, null, 2);
    }

    if (doPreprocess && routeDecision.strategy !== "full-page") {
      const pp = await preprocessForOCR(imageBuf, {
        grayscale: detectedDocType !== "handwriting",
        removeBackground: true,
        enhanceContrast: true,
        sharpen: true,
        docType: detectedDocType,
      });
      imageBuf = Buffer.from(pp.buffer);
      imageMime = pp.mime;
    } else if (doPreprocess) {
      console.error("[extract-fields] Skipping generic preprocessing for full-page route");
    }

    const fieldSpecs: FieldSpec[] = toFieldSpecs(rawFields);

    const config: LayeredExtractionConfig = {
      primaryModel: MODEL,
      ocrModel: useOcrModel ? OCR_MODEL : MODEL,
      primaryBaseUrl: BASE_URL,
      ocrBaseUrl: OCR_BASE_URL || BASE_URL,
      apiKey: API_KEY,
      enableThinking: enableThinking !== undefined ? enableThinking : ENABLE_THINKING,
      enableCE: true,
      enablePostProcess: true,
      docType: detectedDocType,
      forcedStrategy,
      ocrVerify: useOcrModel,
      costPolicy,
      cachePolicy,
      returnCostBreakdown: args.return_cost_breakdown !== false,
      maxUnverifiedRequiredFields: typeof args.max_unverified_required_fields === "number" ? args.max_unverified_required_fields : undefined,
    } as any;

    const result = await layeredExtract(imageBuf, imageMime, fieldSpecs, config);

    return JSON.stringify(result, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- extract_with_verification (v8: routes through layeredExtract) ----
export async function handleExtractVerify(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const path = args.image_path as string;
  const prompt = args.prompt as string;
  const rawRules = args.validation_rules as any[] | undefined;
  const useThinking = args.use_thinking !== false;
  const doPreprocess = args.preprocess !== false;

  if (!path) return JSON.stringify({ error: "image_path required" });
  if (!prompt) return JSON.stringify({ error: "prompt required" });

  try {
    let imageBuf: Buffer = readFileSync(path);
    let imageMime = extToMime(extname(path).toLowerCase());

    const docDetection2 = await detectDocumentType(imageBuf);
    if (doPreprocess) {
      const pp = await preprocessForOCR(imageBuf, {
        grayscale: docDetection2.type !== "handwriting",
        removeBackground: true,
        enhanceContrast: true,
        sharpen: true,
        docType: docDetection2.type,
      });
      imageBuf = Buffer.from(pp.buffer);
      imageMime = pp.mime;
    }

    let jsonResult: Record<string, any> = {};
    let reasoning = "";

    if (useThinking) {
      const client2 = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL, timeout: 300_000 });
      const { thinkingThenStructure } = await import("../extraction/extract-verify-helper.js");
      const ts = await thinkingThenStructure(client2, imageBuf, imageMime, prompt);
      jsonResult = ts.json;
      reasoning = ts.reasoning;
    } else {
      const r = await provider.chat({
        model: MODEL,
        max_tokens: 8192,
        temperature: 0,
        messages: [{
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:${imageMime};base64,${imageBuf.toString("base64")}` } },
            { type: "text", text: `${prompt}\nReturn valid JSON. The JSON keyword must appear.` },
          ],
        }],
        response_format: { type: "json_object" },
      });
      try { jsonResult = JSON.parse(r.text); } catch { jsonResult = { raw: r.text }; }
      reasoning = r.reasoning || "";
    }

    let validationErrors: any[] = [];
    let retried = false;

    if (rawRules && rawRules.length) {
      const { validateFields, retryWithErrorContext } = await import("../extraction/extract-verify-helper.js");
      const fieldValues: Record<string, string> = {};
      for (const [k, v] of Object.entries(jsonResult)) {
        fieldValues[k] = typeof v === "string" ? v : JSON.stringify(v);
      }
      validationErrors = validateFields(fieldValues, rawRules as FieldValidationRule[]);

      if (validationErrors.length > 0) {
        const errorMsg = validationErrors.map((e: any) => `${e.field}: ${e.message}`).join("; ");
        const client2 = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL, timeout: 300_000 });
        const fixed = await retryWithErrorContext(
          client2, imageBuf, imageMime, prompt, errorMsg, JSON.stringify(jsonResult)
        );
        try { jsonResult = JSON.parse(fixed); validationErrors = []; retried = true; } catch { /* keep original */ }
      }
    }

    return JSON.stringify({
      success: true,
      result: jsonResult,
      reasoning,
      validation_errors: validationErrors.length > 0 ? validationErrors : undefined,
      retried,
      mode: useThinking ? "thinking+structured" : "direct",
    }, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- ocr_handwriting (v8: enhanced with detection + dedicated pipeline) ----
export async function handleHandwriting(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const path = args.image_path as string;
  const userPrompt = (args.prompt as string) || "Extract all text from this handwritten document. Preserve original formatting and line breaks. For unclear characters, mark them as [?]. Do not guess.";
  const langHint = (args.language_hint as string) || "";

  if (!path) return JSON.stringify({ error: "image_path required" });

  try {
    let imageBuf: Buffer = readFileSync(path);
    let imageMime = extToMime(extname(path).toLowerCase());

    // v8: Use handwriting-optimized preprocessing
    const pp = await preprocessHandwriting(imageBuf);
    imageBuf = Buffer.from(pp.buffer);
    imageMime = pp.mime;

    // v8: Enhanced handwriting prompt
    let enhancedPrompt = userPrompt;
    if (langHint) {
      enhancedPrompt = `[Language: ${langHint}]\nThis is handwritten ${langHint} text. ` +
        `Read character by character from left to right. ` +
        `Pay attention to: stroke connections, variable stroke widths, irregular spacing. ` +
        `For cursive connections: trace each stroke carefully. ` +
        `For any character you cannot confidently identify, use [?] - do NOT guess. ` +
        `Preserve original formatting, line breaks, and punctuation.\n\n${userPrompt}`;
    } else {
      enhancedPrompt = `This is handwritten text. ` +
        `Read character by character from left to right. ` +
        `Pay attention to: stroke connections, variable stroke widths, irregular spacing. ` +
        `For any character you cannot confidently identify, use [?] - do NOT guess. ` +
        `Preserve original formatting, line breaks, and punctuation.\n\n${userPrompt}`;
    }
    enhancedPrompt = enhancePrompt(enhancedPrompt);

    const r = await provider.chat({
      model: MODEL,
      max_tokens: 16384,
      temperature: 0, // v8: deterministic for handwriting
      messages: [{
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:${imageMime};base64,${imageBuf.toString("base64")}` } },
          { type: "text", text: enhancedPrompt },
        ],
      }],
      vl_high_resolution_images: true, // v8: high-res for handwriting
    });

    return JSON.stringify({
      success: true,
      text: r.text,
      steps: pp.appliedSteps,
      lang_hint: langHint || "auto",
      input_tokens: r.it,
      output_tokens: r.ot,
    }, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}
