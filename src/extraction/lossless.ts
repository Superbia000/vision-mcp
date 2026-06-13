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
  OUTPUT_SCHEMA,
  PRESERVE_ALL,
  RETURN_COST_BREAKDOWN,
  VL_HIGH_RES_ENABLED,
} from "../config/constants.js";
import { detectDocumentType } from "../preprocessing/pipeline.js";
import { estimateImageTokens, summarizeCost } from "../runtime/cost.js";
import { buildLosslessDocumentPrompt } from "./prompts.js";
import { fieldValueLooksValid } from "./router.js";

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
    required: f.required !== false,
  })).filter((f) => f.name || f.labelPattern);
}

export async function extractLosslessDocument(
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
  const prompt = buildLosslessDocumentPrompt(docType, fieldSpecs, pageNumber);
  const estimatedImageTokens = await estimateImageTokens(imageBuffer);

  const response = await provider.chat({
    model: MODEL,
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
    response_format: { type: "json_object" },
  });

  const parsed = parseJsonObject(response.text);
  const page = normalizeLosslessPage(parsed, fieldSpecs, pageNumber);
  const qualityGate = buildQualityGate([page], fieldSpecs, options.maxUnverifiedRequiredFields ?? 0);
  const costBreakdown: CostBreakdownEntry[] = [{
    stage: "lossless_full_page",
    model: MODEL,
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
    finalJson: buildFinalJson([page], fieldSpecs),
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

export function aggregateLosslessPages(
  pageResults: any[],
  rawFields?: any[],
  sourcePath?: string
): LosslessDocumentResult {
  const started = Date.now();
  const fieldSpecs = toFieldSpecs(rawFields);
  const pages: LosslessPage[] = [];
  const errors: string[] = [];
  const costBreakdown: CostBreakdownEntry[] = [];
  let apiCalls = 0;

  for (const result of pageResults) {
    const pageNumber = Number(result?.page || pages.length + 1);
    if (!result?.success) {
      errors.push(`Page ${pageNumber}: ${result?.error || "unknown error"}`);
      pages.push(emptyPage(pageNumber, `Page failed: ${result?.error || "unknown error"}`));
      continue;
    }

    const parsed = parseJsonObject(String(result.text || "{}"));
    const rawPage = parsed?.schema === "lossless_document_v1" ? parsed.pages?.[0] : parsed;
    pages.push(normalizeLosslessPage({ pages: [rawPage] }, fieldSpecs, pageNumber));
    if (Array.isArray(parsed?.costBreakdown)) costBreakdown.push(...parsed.costBreakdown);
    apiCalls += Number(parsed?.stats?.totalApiCalls || 1);
  }

  pages.sort((a, b) => a.page - b.page);
  const qualityGate = buildQualityGate(pages, fieldSpecs, 0);
  return {
    success: qualityGate.passed && errors.length === 0,
    schema: "lossless_document_v1",
    source_path: sourcePath,
    pages,
    finalJson: buildFinalJson(pages, fieldSpecs),
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
