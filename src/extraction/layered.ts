/**
 * Vision-MCP v9: Unified Extraction Engine (simplified)
 *
 * Entry point for all field extraction. Routes based on document type:
 *   scan/photo/handwriting/mixed → full-page OCR + dual-pass annealing (single model)
 *   table                        → L1-L5 layered pipeline (needs bbox for columns)
 *
 * v9 simplifications:
 * - Removed L2 per-field OCR with dedicated model (qwen-vl-ocr)
 * - Removed second-pass correction (qwen3-vl-flash)
 * - Removed L3 cross-validate LLM call (multi-pass voting replaces it)
 * - Post-processing now only does safe normalizations (trim, date separators)
 * - Multi-pass uses temperature annealing (0 → 0.03, early stop)
 */

import OpenAI from "openai";
import sharp from "sharp";
import type { FieldSpec, LayeredExtractionResult, LayeredExtractionConfig } from "../config/types.js";
import {
  ENABLE_CE,
  CE_THRESHOLD,
  ENABLE_POST_PROCESS,
  ENABLE_CROSS_FIELD,
  MULTIPASS_ENABLED,
  MULTIPASS_VOTES,
  L1_TIMEOUT_MS,
  SECOND_PASS_ENABLED,
} from "../config/constants.js";
import type { VisionProvider } from "../providers/base.js";
import { detectDocumentType } from "../preprocessing/pipeline.js";
import { decideStrategy, fieldValueLooksValid, fullPageExtract, multiPassExtract } from "./router.js";
import { QwenProvider } from "../providers/qwen.js";
import { OpenAICompatProvider } from "../providers/openai-compat.js";
import { analyzeLayout } from "./layout.js";
import { cropFieldPreprocessed, extractFieldValue } from "./field-ocr.js";
import { computeFieldConsensusEntropy } from "./cross-validate.js";
import { applyFormatCorrection, validateCrossField } from "./post-process.js";
import { estimateImageTokens, recordVisionCost, summarizeCost } from "../runtime/cost.js";
import { COST_POLICY, CACHE_POLICY, MAX_UNVERIFIED_REQUIRED_FIELDS, RETURN_COST_BREAKDOWN } from "../config/constants.js";

function createProvider(baseUrl: string, apiKey: string): VisionProvider {
  const isQwen = baseUrl.includes("dashscope") || baseUrl.includes("aliyuncs");
  return isQwen ? new QwenProvider(baseUrl, apiKey) : new OpenAICompatProvider(baseUrl, apiKey);
}

export async function layeredExtract(
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  config: LayeredExtractionConfig
): Promise<LayeredExtractionResult> {
  const tStart = Date.now();
  let totalCalls = 0;
  const errors: string[] = [];
  const costBreakdown: any[] = [];
  const routingTrace: Record<string, string[]> = Object.fromEntries(fieldSpecs.map((spec) => [spec.name, []]));
  const costPolicy = config.costPolicy || COST_POLICY;
  const cachePolicy = config.cachePolicy || CACHE_POLICY;
  const maxUnverifiedRequiredFields = config.maxUnverifiedRequiredFields ?? MAX_UNVERIFIED_REQUIRED_FIELDS;
  const returnCostBreakdown = config.returnCostBreakdown ?? RETURN_COST_BREAKDOWN;

  const primaryProvider = config.primaryProviderOverride || createProvider(config.primaryBaseUrl, config.apiKey);
  const ocrProvider = config.ocrProviderOverride || createProvider(config.ocrBaseUrl || config.primaryBaseUrl, config.apiKey);
  const imageBase64 = imageBuffer.toString("base64");
  const pageImageTokens = await estimateImageTokens(imageBuffer);

  // ---- v9: Strategy Routing (auto-detect doc type) ----
  const effectiveDocType = (config as any).docType || (await detectDocumentType(imageBuffer)).type;
  (config as any).docType = effectiveDocType; // cache for downstream
  const decision = decideStrategy(effectiveDocType, (config as any).forcedStrategy);
  console.error(`[layered v9] Strategy: ${decision.strategy} (docType=${effectiveDocType}, reason: ${decision.reason})`);

  // ---- v9: Simplified path for scan/photo/handwriting/mixed ----
  // Single model + dual-pass annealing, no L2 cropping, no second-pass, no character mapping
  if (decision.strategy === "full-page") {
    if (MULTIPASS_ENABLED && MULTIPASS_VOTES > 1) {
      const mp = await multiPassExtract(
        primaryProvider, imageBuffer, imageMime, fieldSpecs, config, MULTIPASS_VOTES
      );
      for (const usage of mp.usage || []) {
        costBreakdown.push({
          stage: usage.stage,
          model: usage.model,
          input_tokens: usage.input_tokens,
          output_tokens: usage.output_tokens,
          estimated_image_tokens: pageImageTokens,
          elapsed_ms: 0,
          cost_policy: costPolicy,
          notes: [`cache_policy=${cachePolicy}`, "full-page image first for same-page cache affinity"],
        });
      }

      let finalJson = mp.finalJson;
      let layout: any = { fields: [], rawText: "", reasoning: "" };
      const uncertainSpecs = fieldsNeedingOcr(fieldSpecs, finalJson);
      if (uncertainSpecs.length > 0 && shouldRunOcrFallback(config)) {
        layout = await locateFieldsForFallback(primaryProvider, config.primaryModel, imageBase64, imageMime, fieldSpecs, costBreakdown, pageImageTokens, costPolicy, cachePolicy);
        totalCalls++;
        const fallback = await runCropOcrFallback(
          ocrProvider,
          imageBuffer,
          imageMime,
          fieldSpecs,
          finalJson,
          layout,
          config,
          routingTrace,
          costBreakdown,
          costPolicy
        );
        finalJson = fallback.finalJson;
        totalCalls += fallback.apiCalls;
      }
      applyFinalReviewGuards(finalJson, fieldSpecs, maxUnverifiedRequiredFields);
      const elapsed = Date.now() - tStart;
      return {
        success: !hasTooManyUnverifiedRequired(finalJson, fieldSpecs, maxUnverifiedRequiredFields),
        layout,
        verifiedFields: {},
        finalJson,
        consensusEntropy: Object.fromEntries(
          Object.entries(mp.consensus).map(([k, v]) => [k, 1 - v.agreement])
        ),
        costBreakdown: returnCostBreakdown ? costBreakdown : undefined,
        routingTrace,
        stats: {
          totalApiCalls: mp.passes.length + totalCalls,
          totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
          elapsedMs: elapsed,
          estimatedImageTokens: pageImageTokens,
        },
      } as any;
    } else {
      const fpResult = await fullPageExtract(
        primaryProvider, imageBuffer, imageMime, fieldSpecs, config
      );
      for (const usage of ((fpResult as any)._visionUsage || [])) {
        costBreakdown.push({
          stage: usage.stage,
          model: usage.model,
          input_tokens: usage.input_tokens,
          output_tokens: usage.output_tokens,
          estimated_image_tokens: pageImageTokens,
          elapsed_ms: 0,
          cost_policy: costPolicy,
          notes: [`cache_policy=${cachePolicy}`, "single deterministic full-page pass"],
        });
      }
      let finalJson = fpResult;
      let layout: any = { fields: [], rawText: "", reasoning: "" };
      const uncertainSpecs = fieldsNeedingOcr(fieldSpecs, finalJson);
      if (uncertainSpecs.length > 0 && shouldRunOcrFallback(config)) {
        layout = await locateFieldsForFallback(primaryProvider, config.primaryModel, imageBase64, imageMime, fieldSpecs, costBreakdown, pageImageTokens, costPolicy, cachePolicy);
        totalCalls++;
        const fallback = await runCropOcrFallback(
          ocrProvider,
          imageBuffer,
          imageMime,
          fieldSpecs,
          finalJson,
          layout,
          config,
          routingTrace,
          costBreakdown,
          costPolicy
        );
        finalJson = fallback.finalJson;
        totalCalls += fallback.apiCalls;
      }
      applyFinalReviewGuards(finalJson, fieldSpecs, maxUnverifiedRequiredFields);
      const elapsed = Date.now() - tStart;
      return {
        success: !hasTooManyUnverifiedRequired(finalJson, fieldSpecs, maxUnverifiedRequiredFields),
        layout,
        verifiedFields: {},
        finalJson,
        costBreakdown: returnCostBreakdown ? costBreakdown : undefined,
        routingTrace,
        stats: {
          totalApiCalls: 1 + totalCalls,
          totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
          elapsedMs: elapsed,
          estimatedImageTokens: pageImageTokens,
        },
      } as any;
    }
  }

  // ---- v9: L1-L5 path for table documents only ----
  // Tables need bbox for column alignment, keep the existing pipeline but skip L2-L4

  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width || 1000;

  // L1: Layout Analysis
  console.error("[layered v9] Starting L1: Layout Analysis (table)...");
  let layout: any;
  try {
    const l1Start = Date.now();
    const l1Promise = analyzeLayout(
      primaryProvider, config.primaryModel, imageBase64, imageMime, fieldSpecs
    );
    const timeoutPromise = new Promise<null>((resolve) =>
      setTimeout(() => resolve(null), L1_TIMEOUT_MS)
    );
    const l1Result = await Promise.race([l1Promise, timeoutPromise]);

    if (l1Result === null) {
      console.error(`[layered v9] L1 timed out - retrying`);
      layout = await analyzeLayout(primaryProvider, config.primaryModel, imageBase64, imageMime, fieldSpecs);
    } else {
      layout = l1Result;
    }
    costBreakdown.push({
      stage: "layout_analysis",
      model: config.primaryModel,
      input_tokens: layout?.usage?.input_tokens,
      output_tokens: layout?.usage?.output_tokens,
      estimated_image_tokens: pageImageTokens,
      elapsed_ms: Date.now() - l1Start,
      cost_policy: costPolicy,
      notes: [`cache_policy=${cachePolicy}`, "coarse layout before crop OCR"],
    });
  } catch (err: any) {
    console.error(`[layered v9] L1 failed: ${err.message}`);
    const l1Start = Date.now();
    layout = await analyzeLayout(primaryProvider, config.primaryModel, imageBase64, imageMime, fieldSpecs);
    costBreakdown.push({
      stage: "layout_analysis_retry",
      model: config.primaryModel,
      input_tokens: layout?.usage?.input_tokens,
      output_tokens: layout?.usage?.output_tokens,
      estimated_image_tokens: pageImageTokens,
      elapsed_ms: Date.now() - l1Start,
      cost_policy: costPolicy,
      notes: [`cache_policy=${cachePolicy}`, "layout retry after timeout/failure"],
    });
  }
  totalCalls++;

  // v9: For table docs, use layout values directly (no L2 per-field OCR, no second-pass)
  const finalJson: Record<string, any> = {};
  for (const spec of fieldSpecs) {
    const layoutField = layout.fields.find(
      (f: any) =>
        f.name.toLowerCase() === spec.name.toLowerCase() ||
        f.label.toLowerCase().includes(spec.labelPattern.toLowerCase())
    );
    finalJson[spec.name] = layoutField?.value
      ? { value: layoutField.value, confidence: layoutField.confidence || "medium", verified: false, source: "L1 layout (table)" }
      : { value: "", confidence: "low", verified: false, error: "no value found" };
    routingTrace[spec.name].push(layoutField?.bbox ? "L1 layout provided bbox" : "L1 layout only; no bbox");
  }

  if (shouldRunOcrFallback(config)) {
    const fallback = await runCropOcrFallback(
      ocrProvider,
      imageBuffer,
      imageMime,
      fieldSpecs,
      finalJson,
      layout,
      config,
      routingTrace,
      costBreakdown,
      costPolicy
    );
    Object.assign(finalJson, fallback.finalJson);
    totalCalls += fallback.apiCalls;
  }

  // CE scoring (keep for info)
  let consensusEntropy: Record<string, number> | undefined;
  if (ENABLE_CE) {
    consensusEntropy = {};
    for (const spec of fieldSpecs) {
      const lf = layout.fields.find((f: any) =>
        f.name.toLowerCase() === spec.name.toLowerCase() ||
        f.label.toLowerCase().includes(spec.labelPattern.toLowerCase())
      );
      const val = lf?.value || "";
      consensusEntropy[spec.name] = val ? 0.1 : 1.0;
    }
  }

  // L5: Post-processing (safe normalizations only)
  const postProcessCorrections: Record<string, { original: string; corrected: string; reason: string }> = {};
  if (ENABLE_POST_PROCESS) {
    for (const spec of fieldSpecs) {
      const entry = finalJson[spec.name];
      if (!entry || !entry.value) continue;
      // v9: Only safe normalizations - no character mapping
      const trimmed = entry.value.trim();
      const normalized = trimmed.replace(/\//g, "-");
      if (normalized !== entry.value) {
        postProcessCorrections[spec.name] = { original: entry.value, corrected: normalized, reason: "normalization" };
        finalJson[spec.name] = { ...entry, value: normalized, post_processed: true };
      }
    }
  }
  applyFinalReviewGuards(finalJson, fieldSpecs, maxUnverifiedRequiredFields);

  const elapsed = Date.now() - tStart;
  console.error(
    `[layered v9] Complete: ${fieldSpecs.length} fields in ${elapsed}ms, ${totalCalls} API calls`
  );

  return {
    success: errors.length === 0 && !hasTooManyUnverifiedRequired(finalJson, fieldSpecs, maxUnverifiedRequiredFields),
    layout,
    verifiedFields: {},
    finalJson,
    errors: errors.length > 0 ? errors : undefined,
    consensusEntropy,
    costBreakdown: returnCostBreakdown ? costBreakdown : undefined,
    routingTrace,
    postProcessCorrections: Object.keys(postProcessCorrections).length > 0 ? postProcessCorrections : undefined,
    stats: {
      totalApiCalls: totalCalls,
      totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
      elapsedMs: elapsed,
      estimatedImageTokens: pageImageTokens,
    },
  };
}

function shouldRunOcrFallback(config: LayeredExtractionConfig): boolean {
  return (config as any).ocrVerify !== false && !!config.ocrModel;
}

function fieldsNeedingOcr(fieldSpecs: FieldSpec[], finalJson: Record<string, any>): FieldSpec[] {
  return fieldSpecs.filter((spec) => {
    const entry = finalJson[spec.name];
    const value = String(entry?.value ?? "");
    return (
      entry?.verified !== true ||
      String(entry?.confidence ?? "").toLowerCase() === "low" ||
      value.includes("[?]") ||
      !fieldValueLooksValid(value, spec)
    );
  });
}

async function locateFieldsForFallback(
  provider: VisionProvider,
  model: string,
  imageBase64: string,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  costBreakdown: any[],
  estimatedImageTokens: number | undefined,
  costPolicy: string,
  cachePolicy: string
) {
  const started = Date.now();
  const layout = await analyzeLayout(provider, model, imageBase64, imageMime, fieldSpecs);
  costBreakdown.push({
    stage: "layout_for_uncertain_fields",
    model,
    input_tokens: (layout as any).usage?.input_tokens,
    output_tokens: (layout as any).usage?.output_tokens,
    estimated_image_tokens: estimatedImageTokens,
    elapsed_ms: Date.now() - started,
    cost_policy: costPolicy,
    notes: [`cache_policy=${cachePolicy}`, "only run because at least one field was not verified"],
  });
  return layout;
}

async function runCropOcrFallback(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  currentJson: Record<string, any>,
  layout: any,
  config: LayeredExtractionConfig,
  routingTrace: Record<string, string[]>,
  costBreakdown: any[],
  costPolicy: string
): Promise<{ finalJson: Record<string, any>; apiCalls: number }> {
  const meta = await sharp(imageBuffer).metadata();
  const imageWidth = meta.width || 1000;
  const imageHeight = meta.height || 1000;
  let apiCalls = 0;
  const finalJson = { ...currentJson };

  for (const spec of fieldsNeedingOcr(fieldSpecs, currentJson)) {
    const layoutField = findLayoutField(layout?.fields || [], spec);
    if (!layoutField?.bbox) {
      routingTrace[spec.name]?.push("skip crop OCR: no bbox from layout");
      finalJson[spec.name] = {
        ...(finalJson[spec.name] || {}),
        verified: false,
        needs_review: true,
        routing_note: "No bbox available for cost-efficient crop OCR; preserved existing value.",
      };
      continue;
    }

    const started = Date.now();
    const crop = await cropFieldPreprocessed(imageBuffer, layoutField.bbox, imageWidth, imageHeight);
    const ocr = await extractFieldValue(provider, config.ocrModel, crop.buffer.toString("base64"), spec, crop.mime, crop.buffer);
    apiCalls++;
    costBreakdown.push({
      stage: `crop_ocr:${spec.name}`,
      model: config.ocrModel,
      input_tokens: ocr.input_tokens,
      output_tokens: ocr.output_tokens,
      estimated_image_tokens: ocr.estimated_image_tokens,
      elapsed_ms: Date.now() - started,
      cost_policy: costPolicy,
      notes: ["field-level crop keeps quality high while avoiding repeated full-page calls"],
    });

    const oldValue = String(currentJson[spec.name]?.value ?? "");
    const newValue = String(ocr.value ?? "").trim();
    const newValid = fieldValueLooksValid(newValue, spec);
    const oldValid = fieldValueLooksValid(oldValue, spec);
    const adoptCrop = !!newValue && (newValid || !oldValid || oldValue.includes("[?]") || !oldValue);
    const chosenValue = adoptCrop ? newValue : oldValue;
    const verified = !!chosenValue && fieldValueLooksValid(chosenValue, spec) && !chosenValue.includes("[?]");

    routingTrace[spec.name]?.push(adoptCrop ? "crop OCR adopted" : "crop OCR kept as candidate; full-page value preserved");
    finalJson[spec.name] = {
      ...(currentJson[spec.name] || {}),
      value: chosenValue,
      confidence: verified ? "high" : (currentJson[spec.name]?.confidence || ocr.confidence || "low"),
      verified,
      needs_review: !verified,
      bbox: layoutField.bbox,
      source: adoptCrop ? "crop OCR fallback" : currentJson[spec.name]?.source || "full-page OCR",
      candidates: [
        { source: currentJson[spec.name]?.source || "current", value: oldValue, valid: oldValid },
        { source: "crop OCR fallback", value: newValue, valid: newValid },
      ],
    };
  }

  return { finalJson, apiCalls };
}

function findLayoutField(fields: any[], spec: FieldSpec): any | undefined {
  const labelPattern = spec.labelPattern.toLowerCase();
  const patterns = labelPattern.split("|").map((p) => p.trim()).filter(Boolean);
  return fields.find((field) => {
    const name = String(field.name || "").toLowerCase();
    const label = String(field.label || "").toLowerCase();
    return name === spec.name.toLowerCase() || patterns.some((p) => label.includes(p) || p.includes(label));
  });
}

function applyFinalReviewGuards(
  finalJson: Record<string, any>,
  fieldSpecs: FieldSpec[],
  maxUnverifiedRequiredFields: number
) {
  for (const spec of fieldSpecs) {
    const entry = finalJson[spec.name] || {};
    const value = String(entry.value ?? "");
    const verified = entry.verified === true && fieldValueLooksValid(value, spec) && !value.includes("[?]");
    finalJson[spec.name] = {
      ...entry,
      verified,
      needs_review: !verified,
      ...(spec.required !== false && !verified ? { required_unverified: true } : {}),
    };
  }
  if (hasTooManyUnverifiedRequired(finalJson, fieldSpecs, maxUnverifiedRequiredFields)) {
    finalJson._quality_gate = {
      passed: false,
      reason: `required unverified fields exceed ${maxUnverifiedRequiredFields}`,
    };
  }
}

function hasTooManyUnverifiedRequired(
  finalJson: Record<string, any>,
  fieldSpecs: FieldSpec[],
  maxUnverifiedRequiredFields: number
): boolean {
  const count = fieldSpecs.filter((spec) => spec.required !== false && finalJson[spec.name]?.verified !== true).length;
  return count > maxUnverifiedRequiredFields;
}
