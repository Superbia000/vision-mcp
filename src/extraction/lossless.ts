import sharp from "sharp";
import type { VisionProvider } from "../providers/base.js";
import type {
  CostBreakdownEntry,
  DocumentType,
  FieldSpec,
  LosslessDocumentResult,
  LosslessFieldCandidate,
  LosslessPage,
} from "../config/types.js";
import {
  MAX_OUTPUT_TOKENS,
  MODEL,
  OCR_MODEL,
  OUTPUT_SCHEMA,
  PRESERVE_ALL,
  RETURN_COST_BREAKDOWN,
  VL_HIGH_RES_ENABLED,
} from "../config/constants.js";
import { detectDocumentType } from "../preprocessing/pipeline.js";
import { estimateImageTokens, summarizeCost } from "../runtime/cost.js";
import { buildLosslessDocumentPrompt } from "./prompts.js";
import { fieldValueLooksValid } from "./router.js";
import { buildUniversalSemanticResult, type SemanticOptions } from "./semantic.js";
import {
  buildFailedUniversalPage,
  extractUniversalDocumentPage,
  mergeUniversalDocumentResults,
} from "./universal-model.js";

type Confidence = "high" | "medium" | "low";

export interface LosslessExtractOptions {
  page?: number;
  sourcePath?: string;
  docType?: DocumentType;
  maxTokens?: number;
  vlHighResolutionImages?: boolean;
  returnCostBreakdown?: boolean;
  preserveAll?: boolean;
  maxUnverifiedRequiredFields?: number;
  costPolicy?: string;
  cachePolicy?: string;
  minPixels?: number;
  maxPixels?: number;
  providerMode?: "auto" | "dashscope_native" | "openai_compat";
  extractionStyle?: "model_first";
  verificationMode?: "off" | "strict";
  returnEvidence?: boolean;
  returnQualityReport?: boolean;
  attentionFields?: any[];
  attentionRules?: any[];
  domainHint?: string;
  semanticMode?: string;
  outputGrain?: string;
  integrationMode?: string;
  extractAllFields?: boolean;
  renderScale?: number;
  maxApiConcurrency?: number;
  renderConcurrency?: number;
  outputDir?: string;
  saveOutputs?: boolean;
  exportFormats?: string[];
  writerMode?: string;
  resumeFrom?: string;
}

export function toFieldSpecs(rawFields?: any[]): FieldSpec[] {
  return (rawFields || []).map((f: any) => ({
    name: String(f.name || "").trim(),
    labelPattern: String(f.label_pattern || f.labelPattern || f.name || "").trim(),
    positionHint: f.position_hint || f.positionHint || undefined,
    formatHint: f.format_hint || f.formatHint || undefined,
    example: f.example || undefined,
    allowedValues: f.allowed_values || f.allowedValues || undefined,
    contextRule: f.context_rule || f.contextRule || undefined,
    required: f.required === true,
  })).filter((f) => f.name || f.labelPattern);
}

export async function extractLosslessDocument(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  rawFields?: any[],
  options: LosslessExtractOptions = {}
): Promise<LosslessDocumentResult> {
  return extractUniversalDocumentPage(provider, imageBuffer, imageMime, rawFields, options);

  const providerMode = options.providerMode || "auto";
  const extractionStyle = options.extractionStyle || "model_first";
  if (
    extractionStyle === "model_first" &&
    providerMode !== "openai_compat" &&
    provider.supportsNativeOcr()
  ) {
    try {
      return await extractModelFirstDocument(provider, imageBuffer, imageMime, rawFields, options);
    } catch (err: any) {
      if (providerMode === "dashscope_native") throw err;
      console.error(`[lossless] Native model-first OCR failed; falling back to OpenAI-compatible path: ${err?.message || err}`);
    }
  }
  return extractPromptLosslessDocument(provider, imageBuffer, imageMime, rawFields, options);
}

async function extractModelFirstDocument(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  rawFields?: any[],
  options: LosslessExtractOptions = {}
): Promise<LosslessDocumentResult> {
  const started = Date.now();
  const fieldSpecs = toFieldSpecs(rawFields);
  const docType = options.docType || (await detectDocumentType(imageBuffer)).type;
  const pageNumber = options.page ?? 1;
  const extractionModel = OCR_MODEL || MODEL;
  const strict = (options.verificationMode || "strict") !== "off";
  const estimatedImageTokens = await estimateImageTokens(imageBuffer);
  const imageQuality = await assessImageQuality(imageBuffer);

  const advanced = await provider.nativeOcr({
    model: extractionModel,
    image: imageBuffer,
    mime: imageMime,
    task: "advanced_recognition",
    max_tokens: options.maxTokens || MAX_OUTPUT_TOKENS,
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
    enable_rotate: true,
    temperature: 0,
  });

  const parsed = await provider.nativeOcr({
    model: extractionModel,
    image: imageBuffer,
    mime: imageMime,
    task: "document_parsing",
    max_tokens: options.maxTokens || MAX_OUTPUT_TOKENS,
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
    enable_rotate: true,
    temperature: 0,
  });

  const wordsInfo = normalizeWordsInfo(advanced.ocrResult?.words_info);
  const advancedText = wordsInfo.map((item) => item.text).filter(Boolean).join("\n") || advanced.text;
  const documentText = parsed.text || "";
  const rawMarkdown = documentText || advancedText;
  const pageIsLowQuality = imageQuality.quality === "low" || (!advancedText.trim() && wordsInfo.length < 3);
  const nativeExtractedFields = fieldSpecs.length
    ? await runModelKeyExtraction(provider, extractionModel, imageBuffer, imageMime, fieldSpecs, options, buildOcrContext(advancedText, documentText))
    : {};
  const reviewExtractedFields = fieldSpecs.length && !pageIsLowQuality && (options.providerMode || "auto") === "auto"
    ? await runModelReviewExtraction(provider, extractionModel, imageBuffer, imageMime, fieldSpecs, options, buildOcrContext(advancedText, documentText))
    : undefined;
  const modelExtractedFields = reviewExtractedFields
    ? mergeModelExtractedFields(nativeExtractedFields, reviewExtractedFields, fieldSpecs, wordsInfo, advancedText, documentText, strict, pageIsLowQuality)
    : nativeExtractedFields;
  const fieldEvidence: Record<string, any> = {};
  const mappedFields: Record<string, any> = {};
  const reviewRequiredFields: any[] = [];
  const finalJson: Record<string, any> = {};

  for (const spec of fieldSpecs) {
    const rawValue = readModelFieldValue(modelExtractedFields, spec.name);
    const value = normalizeModelValue(rawValue);
    const evidence = value ? findEvidence(value, wordsInfo, advancedText, documentText) : { found: false, reason: "model returned empty value" };
    const valid = value ? fieldValueLooksValid(value, spec) : false;
    const safety = value ? fieldSafetyCheck(value, spec, advancedText, documentText) : { passed: false, reasons: [] as string[] };
    const hasUncertainty = value.includes("?") || value.includes("[?]");
    const accepted = !!value && valid && safety.passed && (!strict || (evidence.found && !pageIsLowQuality && !hasUncertainty));
    const reasons = [
      !value ? "missing" : undefined,
      value && !valid ? "format_validation_failed" : undefined,
      value && !safety.passed ? safety.reasons : undefined,
      value && strict && !evidence.found ? "evidence_not_found_in_ocr_text" : undefined,
      value && strict && pageIsLowQuality ? "page_quality_low" : undefined,
      value && strict && hasUncertainty ? "uncertain_character" : undefined,
    ].flat().filter(Boolean) as string[];

    const entry = {
      value,
      normalized_value: value,
      confidence: accepted ? "high" : value && valid && evidence.found ? "medium" : "low",
      verified: accepted,
      needs_review: !accepted,
      validation_state: accepted ? "accepted" : value ? "needs_review" : "missing",
      evidence,
      safety,
      reasons,
      source: "dashscope_native_key_information_extraction",
      required_unverified: spec.required !== false && !accepted,
    };
    mappedFields[spec.name] = entry;
    fieldEvidence[spec.name] = evidence;
    if (accepted) {
      finalJson[spec.name] = entry;
    } else if (spec.required !== false || value) {
      reviewRequiredFields.push({ field: spec.name, value, reasons, evidence });
    }
  }

  const qualityReport = {
    page: pageNumber,
    doc_type: docType,
    image_quality: imageQuality,
    ocr_text_chars: advancedText.length,
    document_text_chars: documentText.length,
    words_info_count: wordsInfo.length,
    hallucination_risk: pageIsLowQuality ||
      allRequiredFieldsMissing(fieldSpecs, reviewRequiredFields) ||
      reviewRequiredFields.some((f) => f.reasons.includes("evidence_not_found_in_ocr_text"))
      ? "high"
      : reviewRequiredFields.length
        ? "medium"
        : "low",
    gate_policy: strict ? "strict_evidence_required" : "format_only",
  };

  const page: LosslessPage = {
    page: pageNumber,
    raw_markdown: rawMarkdown,
    raw_html: "",
    text_items: wordsInfo.map((item) => ({
      text: item.text,
      bbox: item.bbox,
      confidence: "medium",
      source: "dashscope_native_advanced_recognition",
    })),
    tables: [],
    field_candidates: Object.entries(modelExtractedFields)
      .map(([name, rawValue]) => ({
        name,
        value: normalizeModelValue(rawValue),
        confidence: "medium" as Confidence,
        source: modelFieldSource(rawValue),
        needs_review: mappedFields[name]?.needs_review !== false,
      }))
      .filter((item) => item.value),
    mapped_fields: mappedFields,
    unmapped_fields: [],
    orphan_values: [],
    uncertain_tokens: reviewRequiredFields.filter((item) => String(item.value || "").includes("?")),
    review_required: reviewRequiredFields.length > 0 || pageIsLowQuality,
  };

  const qualityGate = buildQualityGate([page], fieldSpecs, options.maxUnverifiedRequiredFields ?? 0);
  const costBreakdown: CostBreakdownEntry[] = [
    toCostEntry("model_first_advanced_recognition", extractionModel, advanced, estimatedImageTokens, started, options),
    toCostEntry("model_first_document_parsing", extractionModel, parsed, estimatedImageTokens, started, options),
  ];
  if ((modelExtractedFields as any).__usage) {
    costBreakdown.push((modelExtractedFields as any).__usage);
    delete (modelExtractedFields as any).__usage;
  }
  if ((nativeExtractedFields as any).__usage) {
    costBreakdown.push((nativeExtractedFields as any).__usage);
    delete (nativeExtractedFields as any).__usage;
  }
  if ((reviewExtractedFields as any)?.__usage) {
    costBreakdown.push((reviewExtractedFields as any).__usage);
    delete (reviewExtractedFields as any).__usage;
  }

  return {
    success: qualityGate.passed && !pageIsLowQuality,
    schema: "lossless_document_v1",
    source_path: options.sourcePath,
    pages: [page],
    finalJson,
    modelExtractedFields,
    fieldEvidence: options.returnEvidence !== false ? fieldEvidence : undefined,
    reviewRequiredFields,
    qualityReport: options.returnQualityReport !== false ? qualityReport : undefined,
    costBreakdown: (options.returnCostBreakdown ?? RETURN_COST_BREAKDOWN) ? costBreakdown : undefined,
    review_required: page.review_required || !qualityGate.passed || pageIsLowQuality,
    quality_gate: pageIsLowQuality
      ? { passed: false, reason: "page quality too low for automatic acceptance" }
      : qualityGate,
    stats: {
      totalApiCalls: fieldSpecs.length ? 3 + (reviewExtractedFields ? 1 : 0) : 2,
      totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
      elapsedMs: Date.now() - started,
      pageCount: 1,
    },
  } as LosslessDocumentResult;
}

async function extractPromptLosslessDocument(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  rawFields?: any[],
  options: LosslessExtractOptions = {}
): Promise<LosslessDocumentResult> {
  const started = Date.now();
  const fieldSpecs = toFieldSpecs(rawFields);
  const docType = options.docType || (await detectDocumentType(imageBuffer)).type;
  const pageNumber = options.page ?? 1;
  const extractionModel = OCR_MODEL || MODEL;
  const prompt = buildLosslessDocumentPrompt(docType, fieldSpecs, pageNumber);
  const estimatedImageTokens = await estimateImageTokens(imageBuffer);

  const response = await provider.chat({
    model: extractionModel,
    max_tokens: options.maxTokens || MAX_OUTPUT_TOKENS,
    temperature: 0,
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${imageMime};base64,${imageBuffer.toString("base64")}` } },
        { type: "text", text: prompt },
      ],
    }],
    vl_high_resolution_images: options.vlHighResolutionImages ?? VL_HIGH_RES_ENABLED,
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
  });

  const parsed = parseJsonObject(response.text);
  const page = normalizeLosslessPage(parsed, fieldSpecs, pageNumber);
  const qualityGate = buildQualityGate([page], fieldSpecs, options.maxUnverifiedRequiredFields ?? 0);
  const costBreakdown: CostBreakdownEntry[] = [{
    stage: "lossless_full_page",
    model: extractionModel,
    input_tokens: response.it,
    output_tokens: response.ot,
    estimated_image_tokens: estimatedImageTokens,
    elapsed_ms: Date.now() - started,
    cost_policy: options.costPolicy || "quality_first",
    notes: [
      `cache_policy=${options.cachePolicy || "auto"}`,
      "full-page lossless parse before requested-field mapping",
      `preserve_all=${options.preserveAll ?? PRESERVE_ALL}`,
    ],
  }];

  return {
    success: qualityGate.passed,
    schema: "lossless_document_v1",
    source_path: options.sourcePath,
    pages: [page],
    finalJson: buildAcceptedFinalJson([page], fieldSpecs),
    costBreakdown: (options.returnCostBreakdown ?? RETURN_COST_BREAKDOWN) ? costBreakdown : undefined,
    review_required: page.review_required || !qualityGate.passed,
    quality_gate: qualityGate,
    stats: {
      totalApiCalls: 1,
      totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
      elapsedMs: Date.now() - started,
      pageCount: 1,
    },
    errors: parsed ? undefined : ["Model response was not valid JSON; preserved raw text in raw_markdown."],
  };
}

async function runModelKeyExtraction(
  provider: VisionProvider,
  extractionModel: string,
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  options: LosslessExtractOptions,
  ocrContext?: string
): Promise<Record<string, any>> {
  const started = Date.now();
  const result = await provider.nativeOcr({
    model: extractionModel,
    image: imageBuffer,
    mime: imageMime,
    task: "key_information_extraction",
    resultSchema: buildModelFirstResultSchema(fieldSpecs),
    text: [
      "Extract only information that is visibly present in the document image.",
      "Return the field value only; do not include printed labels, captions, column headers, or neighboring values.",
      "If one OCR string contains multiple requested values, split them into the matching fields.",
      "Return null for absent, unreadable, or uncertain values.",
      "Do not infer, complete, or fabricate values.",
      ocrContext ? `Model OCR context for cross-checking only:\n${ocrContext}` : undefined,
    ].join(" "),
    max_tokens: options.maxTokens || MAX_OUTPUT_TOKENS,
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
    enable_rotate: true,
    temperature: 0,
  });

  const parsed = result.ocrResult?.kv_result || parseJsonObject(result.text) || {};
  const fields = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? { ...parsed } : {};
  Object.defineProperty(fields, "__usage", {
    value: {
      stage: "model_first_key_information_extraction",
      model: extractionModel,
      input_tokens: result.it,
      output_tokens: result.ot,
      estimated_image_tokens: result.imageTokens,
      elapsed_ms: Date.now() - started,
      cost_policy: options.costPolicy || "quality_first",
      notes: [
        `cache_policy=${options.cachePolicy || "auto"}`,
        "legacy native result_schema extraction",
      ],
    } satisfies CostBreakdownEntry,
    enumerable: false,
    configurable: true,
  });
  return fields;
}

function buildOcrContext(advancedText: string, documentText: string): string {
  const parts = [
    documentText ? `Document parsing text:\n${documentText}` : "",
    advancedText && advancedText !== documentText ? `Advanced recognition text:\n${advancedText}` : "",
  ].filter(Boolean);
  const text = parts.join("\n\n").trim();
  return text.length > 6000 ? `${text.slice(0, 6000)}\n[truncated]` : text;
}

async function runModelReviewExtraction(
  provider: VisionProvider,
  extractionModel: string,
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  options: LosslessExtractOptions,
  ocrContext?: string
): Promise<Record<string, any>> {
  const started = Date.now();
  const fieldLines = fieldSpecs.map((spec) => {
    const readableName = spec.name.replace(/[_-]+/g, " ").trim() || spec.name;
    const semantic = semanticHintForField(spec.name);
    const format = spec.formatHint ? ` Expected format: ${spec.formatHint}.` : "";
    return `- ${spec.name}: ${readableName}.${semantic ? ` ${semantic}` : ""}${format}`;
  }).join("\n");
  const response = await provider.chat({
    model: extractionModel,
    max_tokens: Math.min(options.maxTokens || MAX_OUTPUT_TOKENS, 8192),
    temperature: 0,
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${imageMime};base64,${imageBuffer.toString("base64")}` } },
        {
          type: "text",
          text: [
            "Extract visible document data for the requested schema.",
            "Return strict JSON only, with exactly the requested keys.",
            "Use null for absent, unreadable, or uncertain values.",
            "Return values only; do not include labels, captions, headers, or neighboring values.",
            "Do not infer, complete, calculate, or fabricate values.",
            "Requested fields:",
            fieldLines,
            ocrContext ? `Model OCR context for cross-checking only:\n${ocrContext}` : "",
          ].filter(Boolean).join("\n"),
        },
      ],
    }],
    vl_high_resolution_images: options.vlHighResolutionImages ?? VL_HIGH_RES_ENABLED,
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
  });

  const parsed = parseJsonObject(response.text) || {};
  const fields = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? { ...parsed } : {};
  Object.defineProperty(fields, "__usage", {
    value: {
      stage: "model_first_review_extraction",
      model: extractionModel,
      input_tokens: response.it,
      output_tokens: response.ot,
      estimated_image_tokens: (response as any).imageTokens,
      elapsed_ms: Date.now() - started,
      cost_policy: options.costPolicy || "quality_first",
      notes: [
        `cache_policy=${options.cachePolicy || "auto"}`,
        "model-first Qwen-VL review extraction; no local field filling",
      ],
    } satisfies CostBreakdownEntry,
    enumerable: false,
    configurable: true,
  });
  return fields;
}

function mergeModelExtractedFields(
  nativeFields: Record<string, any>,
  reviewFields: Record<string, any>,
  fieldSpecs: FieldSpec[],
  wordsInfo: { text: string; bbox?: any }[],
  advancedText: string,
  documentText: string,
  strict: boolean,
  pageIsLowQuality: boolean
): Record<string, any> {
  const merged: Record<string, any> = {};
  for (const spec of fieldSpecs) {
    const nativeValue = readModelFieldValue(nativeFields, spec.name);
    const reviewValue = readModelFieldValue(reviewFields, spec.name);
    const nativeAccepted = modelFieldPassesGate(nativeValue, spec, wordsInfo, advancedText, documentText, strict, pageIsLowQuality);
    const reviewAccepted = modelFieldPassesGate(reviewValue, spec, wordsInfo, advancedText, documentText, strict, pageIsLowQuality);
    if (nativeAccepted) {
      merged[spec.name] = nativeValue;
    } else if (reviewAccepted) {
      merged[spec.name] = { value: normalizeModelValue(reviewValue), source: "model_first_review_extraction" };
    } else if (normalizeModelValue(nativeValue)) {
      merged[spec.name] = nativeValue;
    } else if (normalizeModelValue(reviewValue)) {
      merged[spec.name] = { value: normalizeModelValue(reviewValue), source: "model_first_review_extraction" };
    }
  }
  return merged;
}

function modelFieldPassesGate(
  rawValue: any,
  spec: FieldSpec,
  wordsInfo: { text: string; bbox?: any }[],
  advancedText: string,
  documentText: string,
  strict: boolean,
  pageIsLowQuality: boolean
): boolean {
  const value = normalizeModelValue(rawValue);
  if (!value || value.includes("?") || value.includes("[?]")) return false;
  if (!fieldValueLooksValid(value, spec)) return false;
  if (!fieldSafetyCheck(value, spec, advancedText, documentText).passed) return false;
  if (!strict) return true;
  return !pageIsLowQuality && findEvidence(value, wordsInfo, advancedText, documentText).found;
}

function modelFieldSource(rawValue: any): string {
  if (rawValue && typeof rawValue === "object" && rawValue.source) return String(rawValue.source);
  return "dashscope_native_key_information_extraction";
}

function buildModelFirstResultSchema(fieldSpecs: FieldSpec[]): Record<string, string> {
  const schema: Record<string, string> = {};
  for (const spec of fieldSpecs) {
    const readableName = spec.name.replace(/[_-]+/g, " ").trim() || spec.name;
    const hints = [
      `Field: ${readableName}.`,
      semanticHintForField(spec.name),
      spec.formatHint ? `Expected format: ${spec.formatHint}.` : undefined,
      "Return only the value, never the label text.",
      "Extract only if visibly present; return null if absent or unreadable.",
    ].filter(Boolean);
    schema[spec.name] = hints.join(" ");
  }
  return schema;
}

function semanticHintForField(name: string): string | undefined {
  void name;
  return undefined;
}

function allRequiredFieldsMissing(fieldSpecs: FieldSpec[], reviewRequiredFields: any[]): boolean {
  const requiredNames = fieldSpecs.filter((spec) => spec.required !== false).map((spec) => spec.name);
  if (!requiredNames.length) return false;
  const missing = new Set(
    reviewRequiredFields
      .filter((item) => Array.isArray(item.reasons) && item.reasons.includes("missing"))
      .map((item) => item.field)
  );
  return requiredNames.every((name) => missing.has(name));
}

function readModelFieldValue(fields: Record<string, any>, name: string): any {
  if (Object.prototype.hasOwnProperty.call(fields, name)) return fields[name];
  const lower = name.toLowerCase();
  const foundKey = Object.keys(fields).find((key) => key.toLowerCase() === lower);
  return foundKey ? fields[foundKey] : undefined;
}

function normalizeModelValue(value: any): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") {
    const nested = value.value ?? value.text ?? value.raw ?? value.result;
    if (nested !== undefined) return normalizeModelValue(nested);
    return "";
  }
  const text = String(value).trim();
  if (!text) return "";
  if (/^(null|undefined|none|n\/a|na)$/i.test(text)) return "";
  return text;
}

function normalizeWordsInfo(raw: any): { text: string; bbox?: any }[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => ({
      text: String(item?.text || "").trim(),
      bbox: normalizeWordsBbox(item),
    }))
    .filter((item) => item.text);
}

function normalizeWordsBbox(item: any): any {
  const loc = item?.location;
  if (Array.isArray(loc) && loc.length >= 8) {
    const xs = [Number(loc[0]), Number(loc[2]), Number(loc[4]), Number(loc[6])];
    const ys = [Number(loc[1]), Number(loc[3]), Number(loc[5]), Number(loc[7])];
    if ([...xs, ...ys].every((n) => !Number.isNaN(n))) {
      return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
    }
  }
  const rect = item?.rotate_rect;
  if (Array.isArray(rect) && rect.length >= 4) {
    const cx = Number(rect[0]);
    const cy = Number(rect[1]);
    const w = Number(rect[2]);
    const h = Number(rect[3]);
    if ([cx, cy, w, h].every((n) => !Number.isNaN(n))) {
      return { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, angle: Number(rect[4] || 0) };
    }
  }
  return undefined;
}

function findEvidence(
  value: string,
  wordsInfo: { text: string; bbox?: any }[],
  advancedText: string,
  documentText: string
): any {
  const needle = normalizeForEvidence(value);
  if (!needle) return { found: false, reason: "empty normalized value" };

  const line = wordsInfo.find((item) => normalizeForEvidence(item.text).includes(needle));
  if (line) {
    return {
      found: true,
      source: "advanced_recognition.words_info",
      text: line.text,
      bbox: line.bbox,
      match: "normalized_substring",
    };
  }

  if (normalizeForEvidence(advancedText).includes(needle)) {
    return { found: true, source: "advanced_recognition.text", text: value, match: "normalized_substring" };
  }
  if (normalizeForEvidence(documentText).includes(needle)) {
    return { found: true, source: "document_parsing.text", text: value, match: "normalized_substring" };
  }
  return { found: false, reason: "value not found in independent OCR evidence" };
}

function normalizeForEvidence(value: string): string {
  return String(value || "")
    .normalize("NFKC")
    .toUpperCase()
    .replace(/[^\p{L}\p{N}]+/gu, "");
}

function fieldSafetyCheck(
  value: string,
  spec: FieldSpec,
  advancedText: string,
  documentText: string
): { passed: boolean; reasons: string[] } {
  void advancedText;
  void documentText;
  const reasons: string[] = [];
  const compactValue = normalizeForEvidence(value);
  const readableTokens = spec.name
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 4 && !["number", "date", "code", "amount", "total"].includes(token));

  if (readableTokens.some((token) => compactValue.includes(normalizeForEvidence(token)))) {
    reasons.push("label_text_in_value");
  }

  return { passed: reasons.length === 0, reasons };
}

function hasCurrencyTotalEvidence(value: string, currency: "hkd" | "usd", advancedText: string, documentText: string): boolean {
  const needle = normalizeForEvidence(value);
  if (!needle) return false;
  const currencyTokens = currency === "hkd"
    ? ["HKD", "HKDOLLAR", "HONGKONGDOLLAR"]
    : ["USD", "USDOLLAR", "USADOLLAR", "US DOLLAR", "US-DOLLAR"];
  const totalTokens = ["TOTAL", "BALANCE", "PAYABLE", "OURFAVOUR", "OUR FAVOR", "AMOUNTDUE"];
  const lines = `${documentText}\n${advancedText}`.split(/\r?\n/);
  return lines.some((line) => {
    const normalized = normalizeForEvidence(line);
    return normalized.includes(needle) &&
      currencyTokens.some((token) => normalized.includes(normalizeForEvidence(token))) &&
      totalTokens.some((token) => normalized.includes(normalizeForEvidence(token)));
  });
}

async function assessImageQuality(imageBuffer: Buffer): Promise<Record<string, any>> {
  try {
    const meta = await sharp(imageBuffer).metadata();
    const pixels = Number(meta.width || 0) * Number(meta.height || 0);
    const sampled = await sharp(imageBuffer)
      .resize({ width: 512, height: 512, fit: "inside", withoutEnlargement: true })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const data = sampled.data;
    const width = sampled.info.width;
    let sum = 0;
    for (const value of data) sum += value;
    const mean = data.length ? sum / data.length : 0;
    let variance = 0;
    let edgeTotal = 0;
    let edgeCount = 0;
    for (let i = 0; i < data.length; i++) {
      const delta = data[i] - mean;
      variance += delta * delta;
      if ((i + 1) % width !== 0) {
        edgeTotal += Math.abs(data[i] - data[i + 1]);
        edgeCount++;
      }
      if (i + width < data.length) {
        edgeTotal += Math.abs(data[i] - data[i + width]);
        edgeCount++;
      }
    }
    const contrast = data.length ? Math.sqrt(variance / data.length) : 0;
    const edgeMean = edgeCount ? edgeTotal / edgeCount : 0;
    const quality =
      pixels < 300_000 || contrast < 10 || edgeMean < 2
        ? "low"
        : contrast < 18 || edgeMean < 4
          ? "medium"
          : "high";
    return {
      quality,
      width: meta.width,
      height: meta.height,
      pixels,
      contrast: Number(contrast.toFixed(2)),
      edge_mean: Number(edgeMean.toFixed(2)),
    };
  } catch (err: any) {
    return { quality: "unknown", error: err?.message || String(err) };
  }
}

function toCostEntry(
  stage: string,
  model: string,
  response: any,
  estimatedImageTokens: number | undefined,
  started: number,
  options: LosslessExtractOptions
): CostBreakdownEntry {
  return {
    stage,
    model,
    input_tokens: response.it,
    output_tokens: response.ot,
    estimated_image_tokens: response.imageTokens ?? estimatedImageTokens,
    elapsed_ms: Date.now() - started,
    cost_policy: options.costPolicy || "quality_first",
    notes: [
      `cache_policy=${options.cachePolicy || "auto"}`,
      "legacy native model-first task",
    ],
  };
}

export function aggregateLosslessPages(
  pageResults: any[],
  rawFields?: any[],
  sourcePath?: string,
  semanticOptions: SemanticOptions = {}
): LosslessDocumentResult {
  const started = Date.now();
  const fieldSpecs = toFieldSpecs(rawFields);
  const universalResults: LosslessDocumentResult[] = [];
  const pages: LosslessPage[] = [];
  const errors: string[] = [];
  const costBreakdown: CostBreakdownEntry[] = [];
  const modelExtractedFields: Record<string, any> = {};
  const fieldEvidence: Record<string, any> = {};
  const reviewRequiredFields: any[] = [];
  const qualityReports: any[] = [];
  let apiCalls = 0;

  for (const result of pageResults) {
    const pageNumber = Number(result?.page || pages.length + 1);
    if (!result?.success) {
      const message = `Page ${pageNumber}: ${result?.error || "unknown error"}`;
      errors.push(message);
      universalResults.push(buildFailedUniversalPage(pageNumber, message, {
        ...semanticOptions,
        sourcePath,
      }));
      pages.push(emptyPage(pageNumber, `Page failed: ${result?.error || "unknown error"}`));
      continue;
    }

    const parsed = parseJsonObject(String(result.text || "{}"));
    if (parsed?.universal_schema === "universal_document_semantics_v2") {
      universalResults.push(parsed);
      continue;
    }
    const rawPage = parsed?.schema === "lossless_document_v1" ? parsed.pages?.[0] : parsed;
    pages.push(normalizeLosslessPage({ pages: [rawPage] }, fieldSpecs, pageNumber));
    if (parsed?.modelExtractedFields) modelExtractedFields[`page_${pageNumber}`] = parsed.modelExtractedFields;
    if (parsed?.fieldEvidence) fieldEvidence[`page_${pageNumber}`] = parsed.fieldEvidence;
    if (Array.isArray(parsed?.reviewRequiredFields)) {
      reviewRequiredFields.push(...parsed.reviewRequiredFields.map((item: any) => ({ page: pageNumber, ...item })));
    }
    if (parsed?.qualityReport) qualityReports.push(parsed.qualityReport);
    if (Array.isArray(parsed?.costBreakdown)) costBreakdown.push(...parsed.costBreakdown);
    apiCalls += Number(parsed?.stats?.totalApiCalls || 1);
  }

  if (universalResults.length) {
    return mergeUniversalDocumentResults(universalResults, {
      ...semanticOptions,
      sourcePath,
    });
  }

  pages.sort((a, b) => a.page - b.page);
  const qualityGate = buildQualityGate(pages, fieldSpecs, 0);
  const result: LosslessDocumentResult = {
    success: qualityGate.passed && errors.length === 0,
    schema: "lossless_document_v1",
    source_path: sourcePath,
    pages,
    finalJson: buildAcceptedFinalJson(pages, fieldSpecs),
    modelExtractedFields: Object.keys(modelExtractedFields).length ? modelExtractedFields : undefined,
    fieldEvidence: Object.keys(fieldEvidence).length ? fieldEvidence : undefined,
    reviewRequiredFields: reviewRequiredFields.length ? reviewRequiredFields : undefined,
    qualityReport: qualityReports.length ? {
      pages: qualityReports,
      hallucination_risk: qualityReports.some((q) => q?.hallucination_risk === "high")
        ? "high"
        : qualityReports.some((q) => q?.hallucination_risk === "medium")
          ? "medium"
          : "low",
    } : undefined,
    costBreakdown: costBreakdown.length ? costBreakdown : undefined,
    review_required: pages.some((p) => p.review_required) || !qualityGate.passed,
    quality_gate: qualityGate,
    stats: {
      totalApiCalls: apiCalls,
      totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
      elapsedMs: Date.now() - started,
      pageCount: pages.length,
    },
    errors: errors.length ? errors : undefined,
  };
  return buildUniversalSemanticResult(result, { ...semanticOptions, sourcePath });
}

function normalizeLosslessPage(parsed: any, fieldSpecs: FieldSpec[], pageNumber: number): LosslessPage {
  const rawPage = parsed?.pages?.[0] || parsed?.page || parsed || {};
  const legacyFields = Array.isArray(rawPage.fields) ? rawPage.fields : Array.isArray(parsed?.fields) ? parsed.fields : [];
  const candidates = normalizeCandidates([
    ...(Array.isArray(rawPage.field_candidates) ? rawPage.field_candidates : []),
    ...legacyFields,
  ]);
  const mappedFields = mapRequestedFields(rawPage.mapped_fields || {}, candidates, fieldSpecs);
  const mappedCandidateKeys = new Set(
    Object.values(mappedFields)
      .flatMap((entry: any) => Array.isArray(entry?.candidates) ? entry.candidates : [])
      .map((c: any) => candidateKey(c))
  );
  const modelUnmapped = normalizeCandidates(Array.isArray(rawPage.unmapped_fields) ? rawPage.unmapped_fields : []);
  const unmapped = fieldSpecs.length
    ? uniqueCandidates([...modelUnmapped, ...candidates.filter((c) => !mappedCandidateKeys.has(candidateKey(c)))])
    : uniqueCandidates([...modelUnmapped, ...candidates]);
  const uncertainTokens = Array.isArray(rawPage.uncertain_tokens) ? rawPage.uncertain_tokens : [];
  const orphanValues = Array.isArray(rawPage.orphan_values) ? rawPage.orphan_values : [];
  const reviewRequired = hasReview(mappedFields, candidates, uncertainTokens, fieldSpecs);

  return {
    page: Number(rawPage.page || pageNumber),
    raw_markdown: String(rawPage.raw_markdown || rawPage.rawText || parsed?._raw || ""),
    raw_html: rawPage.raw_html ? String(rawPage.raw_html) : "",
    text_items: Array.isArray(rawPage.text_items) ? rawPage.text_items : [],
    tables: Array.isArray(rawPage.tables) ? rawPage.tables : [],
    field_candidates: candidates,
    mapped_fields: mappedFields,
    unmapped_fields: unmapped,
    orphan_values: orphanValues,
    uncertain_tokens: uncertainTokens,
    review_required: reviewRequired,
  };
}

function mapRequestedFields(
  modelMapped: Record<string, any>,
  candidates: LosslessFieldCandidate[],
  fieldSpecs: FieldSpec[]
): Record<string, any> {
  const mapped: Record<string, any> = {};
  for (const spec of fieldSpecs) {
    const modelEntry = modelMapped?.[spec.name];
    const candidateMatches = uniqueCandidates([
      ...candidates.filter((candidate) => candidateMatchesSpec(candidate, spec)),
      ...normalizeCandidates(Array.isArray(modelEntry?.candidates) ? modelEntry.candidates : []),
    ]);
    const modelValue = typeof modelEntry?.value === "string" ? modelEntry.value.trim() : "";
    const candidateValues = candidateMatches.map((c) => c.value).filter(Boolean);
    const values = uniqueStrings([modelValue, ...candidateValues].filter(Boolean));
    const best = chooseBestValue(values, spec);
    const valid = fieldValueLooksValid(best, spec);
    const conflict = values.length > 1;
    const confidence = confidenceFor(modelEntry?.confidence, candidateMatches, valid, conflict);

    mapped[spec.name] = {
      value: best,
      label: modelEntry?.label || candidateMatches[0]?.label || "",
      confidence,
      verified: !!best && valid && confidence === "high" && !conflict,
      needs_review: !best || !valid || confidence !== "high" || conflict,
      conflict,
      candidates: candidateMatches.length
        ? candidateMatches
        : values.map((value) => ({ value, source: "model_mapped", confidence: confidence as Confidence })),
      ...(spec.required !== false && (!best || !valid || conflict) ? { required_unverified: true } : {}),
      ...(!valid && best ? { validation_warning: "value does not match requested field shape" } : {}),
    };
  }
  return mapped;
}

function normalizeCandidates(raw: any[]): LosslessFieldCandidate[] {
  return uniqueCandidates(raw.map((item) => {
    if (typeof item === "string") {
      return { value: item, confidence: "low" as Confidence, source: "model_text", needs_review: true };
    }
    const value = item?.value ?? item?.text ?? item?.raw ?? "";
    const confidence = normalizeConfidence(item?.confidence);
    return {
      name: item?.name ? String(item.name) : undefined,
      label: item?.label ? String(item.label) : undefined,
      value: typeof value === "string" ? value.trim() : String(value ?? "").trim(),
      bbox: normalizeBbox(item?.bbox),
      confidence,
      source: item?.source ? String(item.source) : "full_page",
      needs_review: item?.needs_review === true || confidence !== "high" || String(value).includes("[?]"),
    };
  }).filter((item) => item.value || item.label));
}

function candidateMatchesSpec(candidate: LosslessFieldCandidate, spec: FieldSpec): boolean {
  const haystack = `${candidate.name || ""} ${candidate.label || ""}`.toLowerCase();
  const patterns = [spec.name, spec.labelPattern]
    .flatMap((value) => String(value || "").split("|"))
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  return patterns.some((pattern) => haystack.includes(pattern) || pattern.includes(haystack.trim()));
}

function chooseBestValue(values: string[], spec: FieldSpec): string {
  const valid = values.find((value) => fieldValueLooksValid(value, spec) && !value.includes("[?]"));
  return valid || values[0] || "";
}

function confidenceFor(
  modelConfidence: any,
  candidates: LosslessFieldCandidate[],
  valid: boolean,
  conflict: boolean
): Confidence {
  if (!valid || conflict) return "low";
  const confidence = normalizeConfidence(modelConfidence);
  if (confidence !== "low") return confidence;
  if (candidates.some((c) => c.confidence === "high")) return "high";
  if (candidates.some((c) => c.confidence === "medium")) return "medium";
  return "low";
}

function hasReview(
  mappedFields: Record<string, any>,
  candidates: LosslessFieldCandidate[],
  uncertainTokens: any[],
  fieldSpecs: FieldSpec[]
): boolean {
  if (uncertainTokens.length > 0) return true;
  if (candidates.some((c) => c.needs_review || c.confidence === "low" || c.value.includes("[?]"))) return true;
  return fieldSpecs.some((spec) => spec.required !== false && mappedFields[spec.name]?.verified !== true);
}

function buildFinalJson(pages: LosslessPage[], fieldSpecs: FieldSpec[]): Record<string, any> {
  const finalJson: Record<string, any> = {};
  for (const spec of fieldSpecs) {
    const perPage = pages
      .map((page) => ({ page: page.page, ...(page.mapped_fields[spec.name] || {}) }))
      .filter((entry) => entry.value || entry.candidates?.length);
    const values = uniqueStrings(perPage.map((entry) => String(entry.value || "")).filter(Boolean));
    const best = chooseBestValue(values, spec);
    const conflict = values.length > 1;
    finalJson[spec.name] = {
      value: best,
      confidence: best && fieldValueLooksValid(best, spec) && !conflict ? "high" : "low",
      verified: !!best && fieldValueLooksValid(best, spec) && !conflict,
      needs_review: !best || !fieldValueLooksValid(best, spec) || conflict,
      conflict,
      candidates: perPage,
      source: "lossless_document_v1",
      ...(spec.required !== false && (!best || !fieldValueLooksValid(best, spec) || conflict) ? { required_unverified: true } : {}),
    };
  }
  return finalJson;
}

function buildAcceptedFinalJson(pages: LosslessPage[], fieldSpecs: FieldSpec[]): Record<string, any> {
  const finalJson: Record<string, any> = {};
  for (const spec of fieldSpecs) {
    const accepted = pages
      .map((page) => ({ page: page.page, ...(page.mapped_fields[spec.name] || {}) }))
      .filter((entry) => entry.verified === true && String(entry.value || "").trim());
    const values = uniqueStrings(accepted.map((entry) => String(entry.value || "")).filter(Boolean));
    if (values.length !== 1) continue;
    const chosen = accepted.find((entry) => String(entry.value || "").trim() === values[0]) || accepted[0];
    finalJson[spec.name] = {
      ...chosen,
      value: values[0],
      verified: true,
      needs_review: false,
      source: chosen.source || "lossless_document_v1",
    };
  }
  return finalJson;
}

function buildQualityGate(pages: LosslessPage[], fieldSpecs: FieldSpec[], maxUnverifiedRequiredFields: number) {
  const unverified = fieldSpecs.filter((spec) => {
    if (spec.required === false) return false;
    return !pages.some((page) => page.mapped_fields[spec.name]?.verified === true);
  }).length;
  if (unverified > maxUnverifiedRequiredFields) {
    return { passed: false, reason: `required unverified fields exceed ${maxUnverifiedRequiredFields}` };
  }
  if (!PRESERVE_ALL && OUTPUT_SCHEMA !== "lossless_document_v1") {
    return { passed: false, reason: "lossless preservation disabled by config" };
  }
  return { passed: true };
}

function emptyPage(page: number, error: string): LosslessPage {
  return {
    page,
    raw_markdown: "",
    raw_html: "",
    text_items: [],
    tables: [],
    field_candidates: [],
    mapped_fields: {},
    unmapped_fields: [],
    orphan_values: [],
    uncertain_tokens: [],
    review_required: true,
  };
}

function parseJsonObject(text: string): any | null {
  try {
    return JSON.parse(text || "{}");
  } catch {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch {
        return { _raw: text };
      }
    }
    return { _raw: text };
  }
}

function normalizeConfidence(value: any): Confidence {
  const text = String(value || "").toLowerCase();
  if (text === "high" || text === "medium" || text === "low") return text;
  return "low";
}

function normalizeBbox(value: any): any {
  if (!value || typeof value !== "object") return undefined;
  const x1 = Number(value.x1 ?? value.x ?? 0);
  const y1 = Number(value.y1 ?? value.y ?? 0);
  const x2 = Number(value.x2 ?? (value.x !== undefined && value.w !== undefined ? Number(value.x) + Number(value.w) : undefined));
  const y2 = Number(value.y2 ?? (value.y !== undefined && value.h !== undefined ? Number(value.y) + Number(value.h) : undefined));
  if ([x1, y1, x2, y2].some((n) => Number.isNaN(n))) return undefined;
  return { x1, y1, x2, y2 };
}

function uniqueCandidates(candidates: LosslessFieldCandidate[]): LosslessFieldCandidate[] {
  const seen = new Set<string>();
  const out: LosslessFieldCandidate[] = [];
  for (const candidate of candidates) {
    const key = candidateKey(candidate);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(candidate);
  }
  return out;
}

function candidateKey(candidate: any): string {
  return [
    String(candidate?.name || "").toLowerCase(),
    String(candidate?.label || "").toLowerCase(),
    String(candidate?.value || "").toLowerCase(),
  ].join("|");
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.map((v) => v.trim()).filter(Boolean))];
}
