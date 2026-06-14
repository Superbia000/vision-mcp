import { basename } from "path";
import { jsonrepair } from "jsonrepair";
import type { VisionProvider } from "../providers/base.js";
import type {
  CostBreakdownEntry,
  LosslessDocumentResult,
  LosslessFieldCandidate,
  LosslessPage,
} from "../config/types.js";
import {
  MAX_OUTPUT_TOKENS,
  MODEL,
  RETURN_COST_BREAKDOWN,
  VL_HIGH_RES_ENABLED,
} from "../config/constants.js";
import { estimateImageTokens, summarizeCost } from "../runtime/cost.js";
import { writeUniversalPageCheckpoint } from "../output/universal-writer.js";

type Confidence = "high" | "medium" | "low";

export interface UniversalDocumentOptions {
  page?: number;
  sourcePath?: string;
  maxTokens?: number;
  vlHighResolutionImages?: boolean;
  returnCostBreakdown?: boolean;
  costPolicy?: string;
  cachePolicy?: string;
  minPixels?: number;
  maxPixels?: number;
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
  writerMode?: string;
  outputDir?: string;
  saveOutputs?: boolean;
  exportFormats?: string[];
  resumeFrom?: string;
}

interface AttentionHint {
  name: string;
  aliases: string[];
  required: boolean;
  original: any;
}

interface NormalizedPage {
  rawPage: any;
  semanticPage: any;
  losslessPage: LosslessPage;
}

export async function extractUniversalDocumentPage(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  rawAttentionFields: any[] | undefined,
  options: UniversalDocumentOptions = {},
): Promise<LosslessDocumentResult> {
  const started = Date.now();
  const pageNumber = options.page ?? 1;
  const attentionHints = normalizeAttentionHints([
    ...(Array.isArray(rawAttentionFields) ? rawAttentionFields : []),
    ...(Array.isArray(options.attentionFields) ? options.attentionFields : []),
  ]);
  const estimatedImageTokens = await estimateImageTokens(imageBuffer);
  const prompt = buildUniversalDocumentPrompt(pageNumber, attentionHints, options);

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
    min_pixels: options.minPixels,
    max_pixels: options.maxPixels,
  });

  const parsed = parseUniversalModelJson(response.text);
  const normalized = normalizeModelPage(parsed, response.text, pageNumber, options, attentionHints);
  const costBreakdown: CostBreakdownEntry[] = [{
    stage: "universal_qwen3_vl_page_understanding",
    model: MODEL,
    input_tokens: response.it,
    output_tokens: response.ot,
    estimated_image_tokens: estimatedImageTokens,
    elapsed_ms: Date.now() - started,
    cost_policy: options.costPolicy || "quality_first",
    notes: [
      `cache_policy=${options.cachePolicy || "auto"}`,
      "single visual model universal OCR/KIE; attention fields are hints only",
    ],
  }];

  const reviewIssues = normalized.semanticPage.review_issues || [];
  const result = buildUniversalResult({
    sourcePath: options.sourcePath,
    pages: [normalized.losslessPage],
    rawPages: [normalized.rawPage],
    semanticPages: [normalized.semanticPage],
    costBreakdown,
    errors: parsed ? undefined : ["Model response was not valid JSON; raw response preserved for review."],
    elapsedMs: Date.now() - started,
    apiCalls: 1,
    reviewRequired: normalized.losslessPage.review_required || reviewIssues.length > 0 || !parsed,
    options,
  });

  if (options.saveOutputs !== false && options.outputDir && options.writerMode !== "none") {
    try {
      const artifacts = writeUniversalPageCheckpoint(result, {
        outputDir: options.outputDir,
        sourcePath: options.sourcePath,
        writerMode: options.writerMode,
      });
      result.artifacts = [...(result.artifacts || []), ...artifacts];
    } catch (err: any) {
      result.review_issues = [
        ...(result.review_issues || []),
        {
          severity: "medium",
          type: "checkpoint_write_failed",
          message: err?.message || String(err),
          source: normalized.semanticPage.source,
        },
      ];
      result.review_required = true;
    }
  }

  return result;
}

export function mergeUniversalDocumentResults(
  pageResults: LosslessDocumentResult[],
  options: UniversalDocumentOptions = {},
): LosslessDocumentResult {
  const started = Date.now();
  const pages = pageResults.flatMap((result) => result.pages || []).sort((a, b) => a.page - b.page);
  const rawPages = pageResults.flatMap((result) => result.raw_pages || []).sort((a, b) => Number(a?.source?.page || 0) - Number(b?.source?.page || 0));
  const semanticPages = pageResults.flatMap((result) => result.semantic_pages || []).sort((a, b) => Number(a?.source?.page || 0) - Number(b?.source?.page || 0));
  const costBreakdown = pageResults.flatMap((result) => result.costBreakdown || []);
  const errors = pageResults.flatMap((result) => result.errors || []);
  const artifacts = pageResults.flatMap((result) => result.artifacts || []);

  return buildUniversalResult({
    sourcePath: options.sourcePath || pageResults.find((result) => result.source_path)?.source_path,
    pages,
    rawPages,
    semanticPages,
    costBreakdown,
    errors: errors.length ? errors : undefined,
    elapsedMs: Date.now() - started + pageResults.reduce((sum, result) => sum + Number(result.stats?.elapsedMs || 0), 0),
    apiCalls: pageResults.reduce((sum, result) => sum + Number(result.stats?.totalApiCalls || 0), 0),
    reviewRequired: pageResults.some((result) => result.review_required),
    artifacts,
    options,
  });
}

export function buildFailedUniversalPage(
  page: number,
  error: string,
  options: UniversalDocumentOptions = {},
): LosslessDocumentResult {
  const source = buildSource(page, options);
  const issue = {
    severity: "high",
    type: "page_extraction_failed",
    message: error,
    source,
  };
  const rawPage = {
    source,
    raw_content: {
      raw_markdown: "",
      raw_html: "",
      text_items: [],
      tables: [],
      uncertain_tokens: [],
      model_raw_response: "",
    },
  };
  const semanticPage = {
    source,
    document_classification: {
      type: "unknown_document",
      confidence: "low",
      reason: "Page extraction failed before model classification.",
      evidence: [],
      needs_review: true,
    },
    raw_content: rawPage.raw_content,
    field_candidates: [],
    attention_field_matches: normalizeAttentionHints(options.attentionFields).map((hint) => ({
      name: hint.name,
      aliases: hint.aliases,
      status: "not_evaluated",
      matched: false,
      candidates: [],
      needs_review: hint.required,
    })),
    unmapped_fields: [],
    orphan_values: [],
    entities: [],
    relationships: [],
    review_issues: [issue],
  };

  return buildUniversalResult({
    sourcePath: options.sourcePath,
    pages: [emptyLosslessPage(page, error)],
    rawPages: [rawPage],
    semanticPages: [semanticPage],
    errors: [error],
    elapsedMs: 0,
    apiCalls: 0,
    reviewRequired: true,
    options,
  });
}

function buildUniversalDocumentPrompt(
  pageNumber: number,
  attentionHints: AttentionHint[],
  options: UniversalDocumentOptions,
): string {
  const attentionBlock = attentionHints.length
    ? JSON.stringify(attentionHints.map((hint) => ({
        name: hint.name,
        aliases: hint.aliases,
        required: hint.required,
      })), null, 2)
    : "[]";
  const rulesBlock = Array.isArray(options.attentionRules) && options.attentionRules.length
    ? JSON.stringify(options.attentionRules, null, 2)
    : "[]";

  return [
    "You are performing universal document understanding for one page image.",
    "Use the visual page itself as the source of truth. Do not use a business template, domain-specific field list, or prior assumption.",
    "The caller may provide attention hints. They are priority review targets only, never a whitelist. You must still extract every visible field, table, entity, relationship, unknown field, and orphan value.",
    "",
    "Method:",
    "1. Read the entire page globally before naming fields.",
    "2. Preserve all visible text, numbers, dates, codes, handwriting, stamps, table cells, headers, footers, isolated values, and uncertain tokens.",
    "3. Infer field names from visible labels, nearby text, table headers, row/column relations, layout, and repeated patterns on this page.",
    "4. Keep both the exact visible label and your model-derived field name. Never overwrite the original text.",
    "5. If a label/value is visible but cannot be named reliably, put it in unmapped_fields.",
    "6. If an important value has no clear label, put it in orphan_values.",
    "7. Classify the document only from visible evidence on this page. A domain hint may explain vocabulary but must not decide classification.",
    "8. For every important value, include evidence: page number plus raw text, bbox if visible, table id/row/column, or nearby text.",
    "9. If text is blurred, occluded, handwritten, contradictory, or low confidence, mark needs_review and add review_issues. Do not complete missing content.",
    "10. Do not normalize away the original. Put normalized values only in value_normalized while preserving value_original.",
    "",
    `Page number: ${pageNumber}`,
    `Domain hint: ${options.domainHint || "auto"} (attention only, not classification authority)`,
    `Semantic mode: ${options.semanticMode || "auto"}`,
    `Output grain: ${options.outputGrain || "auto"}`,
    `Integration mode: ${options.integrationMode || "none"}`,
    `Extract all fields: ${options.extractAllFields !== false}`,
    "Caller attention fields:",
    attentionBlock,
    "Caller attention rules:",
    rulesBlock,
    "",
    "Return strict JSON only. Use this schema:",
    JSON.stringify({
      schema: "universal_document_semantics_v2",
      source: {
        page: pageNumber,
      },
      document_classification: {
        type: "model-derived document type from visible evidence",
        confidence: "high|medium|low",
        reason: "short visible-evidence reason",
        evidence: ["raw visible text or layout evidence"],
        needs_review: false,
      },
      raw_content: {
        raw_markdown: "all visible text and tables in reading order",
        raw_html: "",
        text_items: [{ text: "visible line or token", bbox: { x1: 0, y1: 0, x2: 999, y2: 999 }, confidence: "high|medium|low" }],
        tables: [{ table_id: "table_1", bbox: { x1: 0, y1: 0, x2: 999, y2: 999 }, rows: [["cell text"]] }],
        uncertain_tokens: [{ text: "[?]", context: "nearby visible text", bbox: { x1: 0, y1: 0, x2: 999, y2: 999 } }],
      },
      field_candidates: [{
        label_original: "exact visible label or empty string",
        field_name_model: "model-derived field name",
        value_original: "exact visible value",
        value_normalized: "optional normalized value",
        value_type_model: "model-derived value type",
        confidence: "high|medium|low",
        evidence: { page: pageNumber, raw_text: "visible evidence", bbox: { x1: 0, y1: 0, x2: 999, y2: 999 } },
        attention_match: null,
        needs_review: false,
      }],
      attention_field_matches: [{
        name: "caller attention field",
        status: "matched|not_found|ambiguous",
        matched: true,
        candidates: [],
        needs_review: false,
      }],
      unmapped_fields: [{ label_original: "visible unknown label", value_original: "visible value", confidence: "high|medium|low", evidence: {} }],
      orphan_values: [{ value_original: "visible value without clear label", value_type_model: "model-derived type", confidence: "high|medium|low", evidence: {} }],
      entities: [{ entity_id: "optional", type: "model-derived entity type", name: "entity name/value", value: "entity value", confidence: "high|medium|low", evidence: {} }],
      relationships: [{ relationship_id: "optional", type: "model-derived relationship type", from: "entity/field id", to: "entity/field id", confidence: "high|medium|low", evidence: {} }],
      integrated_records: [{ record_id: "optional", record_type: "model-derived record type", fields: {}, source_pages: [pageNumber], confidence: "high|medium|low" }],
      review_issues: [{ severity: "high|medium|low", type: "issue type", message: "what requires review", evidence: {} }],
    }, null, 2),
  ].join("\n");
}

function normalizeModelPage(
  parsed: any,
  rawResponse: string,
  pageNumber: number,
  options: UniversalDocumentOptions,
  attentionHints: AttentionHint[],
): NormalizedPage {
  const modelPage = parsed?.pages?.[0] || parsed?.semantic_page || parsed || {};
  const source = buildSource(pageNumber, options);
  const rawContent = normalizeRawContent(modelPage, rawResponse);
  const fieldCandidates = normalizeFieldCandidates(modelPage.field_candidates, pageNumber);
  const attentionMatches = normalizeAttentionMatches(modelPage.attention_field_matches, attentionHints);
  const unmappedFields = normalizeGenericArray(modelPage.unmapped_fields).map((item) => ({
    ...item,
    page_number: item.page_number ?? pageNumber,
    needs_review: item.needs_review !== false,
  }));
  const orphanValues = normalizeGenericArray(modelPage.orphan_values).map((item) => ({
    ...item,
    page_number: item.page_number ?? pageNumber,
    needs_review: item.needs_review !== false,
  }));
  const documentClassification = normalizeClassification(modelPage.document_classification, parsed, fieldCandidates, source);
  const entities = normalizeEntities(modelPage.entities, source);
  const relationships = normalizeRelationships(modelPage.relationships, source);
  const reviewIssues = normalizeReviewIssues(modelPage.review_issues, source);

  for (const candidate of fieldCandidates) {
    if (!hasEvidence(candidate)) {
      candidate.needs_review = true;
      reviewIssues.push({
        severity: "medium",
        type: "field_missing_evidence",
        message: `Field has no explicit evidence: ${candidate.field_name_model || candidate.label_original || "unnamed field"}`,
        field_name_model: candidate.field_name_model,
        source,
      });
    }
  }
  for (const hint of attentionHints) {
    if (!attentionMatches.some((match) => match.name === hint.name)) {
      attentionMatches.push({
        name: hint.name,
        aliases: hint.aliases,
        status: "not_found",
        matched: false,
        candidates: [],
        needs_review: hint.required,
        reason: "Model did not return a match for this attention hint.",
      });
    }
  }
  if (!fieldCandidates.length) {
    reviewIssues.push({
      severity: "high",
      type: "no_field_candidates",
      message: "The model did not return any field candidates for this page.",
      source,
    });
  }
  if (!entities.length) {
    reviewIssues.push({
      severity: "low",
      type: "no_entities_returned",
      message: "The model did not return entities; fields and raw content are still preserved.",
      source,
    });
  }

  const semanticPage = {
    source,
    document_classification: documentClassification,
    raw_content: rawContent,
    field_candidates: fieldCandidates,
    attention_field_matches: attentionMatches,
    unmapped_fields: unmappedFields,
    orphan_values: orphanValues,
    entities,
    relationships,
    integrated_records: normalizeGenericArray(modelPage.integrated_records),
    review_issues: reviewIssues,
  };

  const rawPage = {
    source,
    raw_content: rawContent,
  };

  const losslessPage: LosslessPage = {
    page: pageNumber,
    raw_markdown: rawContent.raw_markdown,
    raw_html: rawContent.raw_html,
    text_items: rawContent.text_items,
    tables: rawContent.tables,
    field_candidates: fieldCandidates.map(fieldToLosslessCandidate),
    mapped_fields: buildAttentionMappedFields(attentionMatches),
    unmapped_fields: unmappedFields.map(genericToLosslessCandidate),
    orphan_values: orphanValues,
    uncertain_tokens: rawContent.uncertain_tokens,
    review_required: reviewIssues.some((issue) => issue.severity !== "low") ||
      fieldCandidates.some((field) => field.needs_review === true) ||
      attentionMatches.some((match) => match.needs_review === true),
  };

  return { rawPage, semanticPage, losslessPage };
}

function normalizeRawContent(modelPage: any, rawResponse: string): any {
  const rawContent = modelPage.raw_content || modelPage.rawContent || {};
  const rawMarkdown = stringifyCell(rawContent.raw_markdown ?? modelPage.raw_markdown ?? modelPage.rawText ?? modelPage.text ?? "");
  return {
    raw_markdown: rawMarkdown,
    raw_html: stringifyCell(rawContent.raw_html ?? modelPage.raw_html ?? ""),
    text_items: normalizeTextItems(rawContent.text_items ?? rawContent.text_lines ?? modelPage.text_items),
    tables: normalizeTables(rawContent.tables ?? modelPage.tables),
    uncertain_tokens: normalizeGenericArray(rawContent.uncertain_tokens ?? modelPage.uncertain_tokens),
    model_raw_response: rawResponse,
  };
}

function normalizeClassification(classification: any, parsed: any, fields: any[], source: any): any {
  const cls = classification && typeof classification === "object" ? classification : {};
  const type = stringifyCell(cls.type ?? parsed?.document_type ?? "document");
  return {
    type: type || "document",
    confidence: normalizeConfidence(cls.confidence),
    reason: stringifyCell(cls.reason ?? cls.rationale ?? ""),
    evidence: Array.isArray(cls.evidence) ? cls.evidence : [],
    needs_review: cls.needs_review === true || !classification || !fields.length || normalizeConfidence(cls.confidence) === "low",
    source,
  };
}

function normalizeFieldCandidates(raw: any, pageNumber: number): any[] {
  return normalizeGenericArray(raw).map((item, index) => {
    const labelOriginal = stringifyCell(item.label_original ?? item.label ?? "");
    const fieldNameModel = stringifyCell(item.field_name_model ?? item.name ?? item.field ?? labelOriginal);
    const valueOriginal = stringifyCell(item.value_original ?? item.value ?? item.text ?? "");
    return {
      field_id: stringifyCell(item.field_id ?? `field_p${pageNumber}_${index + 1}`),
      source_pdf: stringifyCell(item.source_pdf ?? item.source_path ?? ""),
      page_number: Number(item.page_number ?? item.page ?? pageNumber),
      label_original: labelOriginal,
      field_name_model: fieldNameModel,
      name: fieldNameModel,
      label: labelOriginal,
      value_original: valueOriginal,
      value: valueOriginal,
      value_normalized: stringifyCell(item.value_normalized ?? item.normalized_value ?? valueOriginal),
      value_type_model: stringifyCell(item.value_type_model ?? item.type ?? ""),
      confidence: normalizeConfidence(item.confidence),
      evidence: normalizeEvidence(item.evidence ?? { page: pageNumber, raw_text: valueOriginal, bbox: item.bbox }),
      attention_match: item.attention_match ?? null,
      bbox: normalizeBbox(item.bbox ?? item.evidence?.bbox),
      needs_review: item.needs_review === true || normalizeConfidence(item.confidence) === "low",
    };
  }).filter((item) => item.value_original || item.label_original || item.field_name_model);
}

function normalizeAttentionMatches(raw: any, attentionHints: AttentionHint[]): any[] {
  const modelMatches = normalizeGenericArray(raw).map((item) => {
    const status = stringifyCell(item.status || (item.matched ? "matched" : "not_found")) || "not_found";
    return {
      name: stringifyCell(item.name ?? item.field ?? ""),
      aliases: Array.isArray(item.aliases) ? item.aliases.map(stringifyCell).filter(Boolean) : [],
      status,
      matched: item.matched === true || status === "matched",
      candidates: Array.isArray(item.candidates) ? item.candidates : [],
      needs_review: item.needs_review === true || status !== "matched",
      reason: stringifyCell(item.reason ?? ""),
    };
  }).filter((item) => item.name);

  for (const hint of attentionHints) {
    const existing = modelMatches.find((item) => item.name === hint.name);
    if (existing) {
      existing.aliases = existing.aliases.length ? existing.aliases : hint.aliases;
      existing.needs_review = existing.needs_review || (hint.required && !existing.matched);
    }
  }
  return modelMatches;
}

function normalizeEntities(raw: any, source: any): any[] {
  return normalizeGenericArray(raw).map((item, index) => ({
    entity_id: stringifyCell(item.entity_id ?? item.id ?? `ent_p${source.page}_${index + 1}`),
    id: stringifyCell(item.id ?? item.entity_id ?? `ent_p${source.page}_${index + 1}`),
    type: stringifyCell(item.type ?? "entity"),
    name: stringifyCell(item.name ?? item.label ?? item.value ?? ""),
    value: stringifyCell(item.value ?? item.name ?? ""),
    confidence: normalizeConfidence(item.confidence),
    evidence: normalizeEvidence(item.evidence ?? {}),
    source: item.source || source,
    needs_review: item.needs_review === true || normalizeConfidence(item.confidence) === "low",
  })).filter((item) => item.name || item.value);
}

function normalizeRelationships(raw: any, source: any): any[] {
  return normalizeGenericArray(raw).map((item, index) => ({
    relationship_id: stringifyCell(item.relationship_id ?? item.id ?? `rel_p${source.page}_${index + 1}`),
    id: stringifyCell(item.id ?? item.relationship_id ?? `rel_p${source.page}_${index + 1}`),
    type: stringifyCell(item.type ?? "related_to"),
    from: stringifyCell(item.from ?? item.source_id ?? ""),
    to: stringifyCell(item.to ?? item.target_id ?? ""),
    confidence: normalizeConfidence(item.confidence),
    evidence: normalizeEvidence(item.evidence ?? {}),
    source: item.source || source,
    needs_review: item.needs_review === true || normalizeConfidence(item.confidence) === "low",
  })).filter((item) => item.from || item.to || item.type);
}

function normalizeReviewIssues(raw: any, source: any): any[] {
  return normalizeGenericArray(raw).map((item) => ({
    severity: ["high", "medium", "low"].includes(String(item.severity)) ? String(item.severity) : "medium",
    type: stringifyCell(item.type ?? "model_review_issue"),
    message: stringifyCell(item.message ?? item.reason ?? ""),
    evidence: normalizeEvidence(item.evidence ?? {}),
    source: item.source || source,
  })).filter((item) => item.message || item.type);
}

function buildUniversalResult(args: {
  sourcePath?: string;
  pages: LosslessPage[];
  rawPages: any[];
  semanticPages: any[];
  costBreakdown?: CostBreakdownEntry[];
  errors?: string[];
  elapsedMs: number;
  apiCalls: number;
  reviewRequired: boolean;
  artifacts?: any[];
  options: UniversalDocumentOptions;
}): LosslessDocumentResult {
  const detectedDocuments = buildDetectedDocuments(args.semanticPages);
  const entities = reindex(args.semanticPages.flatMap((page) => page.entities || []), "ent");
  const relationships = reindex(args.semanticPages.flatMap((page) => page.relationships || []), "rel");
  const integratedRecords = buildIntegratedRecords(args.semanticPages);
  const reviewIssues = args.semanticPages.flatMap((page) => page.review_issues || []);
  const finalJson = buildFinalJson(args.semanticPages);
  const hasErrors = !!args.errors?.length;
  const costBreakdown = args.costBreakdown || [];

  return {
    success: !hasErrors,
    schema: "lossless_document_v1",
    source_path: args.sourcePath,
    pages: args.pages,
    finalJson,
    universal_schema: "universal_document_semantics_v2",
    extraction_policy: {
      preserve_all: true,
      extract_all_fields: args.options.extractAllFields !== false,
      attention_fields_are_hints_only: true,
      single_visual_model: true,
      model: MODEL,
      domain_hint: args.options.domainHint || "auto",
      semantic_mode: args.options.semanticMode || "auto",
      output_grain: args.options.outputGrain || "auto",
      integration_mode: args.options.integrationMode || "none",
      max_api_concurrency: args.options.maxApiConcurrency,
      render_concurrency: args.options.renderConcurrency,
      writer_mode: args.options.writerMode,
      generated_at: formatDateTime(new Date()),
    },
    raw_pages: args.rawPages,
    semantic_pages: args.semanticPages,
    detected_documents: detectedDocuments,
    entities,
    relationships,
    integrated_records: integratedRecords,
    review_issues: reviewIssues,
    artifacts: args.artifacts,
    costBreakdown: (args.options.returnCostBreakdown ?? RETURN_COST_BREAKDOWN) ? costBreakdown : undefined,
    review_required: args.reviewRequired || reviewIssues.some((issue) => issue.severity !== "low"),
    quality_gate: hasErrors ? { passed: false, reason: args.errors?.join("; ") } : { passed: true },
    stats: {
      totalApiCalls: args.apiCalls,
      totalTokens: summarizeCost(costBreakdown).total_input_tokens || 0,
      elapsedMs: args.elapsedMs,
      pageCount: args.pages.length,
    },
    errors: args.errors,
  };
}

function buildSource(page: number, options: UniversalDocumentOptions): any {
  const sourcePath = options.sourcePath || "";
  return {
    source_path: sourcePath,
    source_file: sourcePath ? basename(sourcePath) : "",
    page,
    model: MODEL,
    extracted_at: formatDateTime(new Date()),
    render: {
      render_scale: options.renderScale,
      min_pixels: options.minPixels,
      max_pixels: options.maxPixels,
    },
  };
}

function buildDetectedDocuments(semanticPages: any[]): any[] {
  return semanticPages.map((page, index) => ({
    document_id: `doc_${String(index + 1).padStart(5, "0")}`,
    source_file: page.source?.source_file || "",
    document_type: page.document_classification?.type || "document",
    confidence: page.document_classification?.confidence || "low",
    pages: [page.source?.page].filter((value: any) => value !== undefined),
    reason: page.document_classification?.reason || "",
    evidence: page.document_classification?.evidence || [],
    needs_review: page.document_classification?.needs_review === true,
  }));
}

function buildIntegratedRecords(semanticPages: any[]): any[] {
  const records: any[] = [];
  for (const page of semanticPages) {
    const modelRecords = normalizeGenericArray(page.integrated_records);
    if (modelRecords.length) {
      records.push(...modelRecords.map((record, index) => ({
        record_id: stringifyCell(record.record_id ?? `page_${page.source?.page}_record_${index + 1}`),
        record_type: stringifyCell(record.record_type ?? page.document_classification?.type ?? "document_record"),
        source_files: record.source_files ?? [page.source?.source_file].filter(Boolean),
        source_pages: record.source_pages ?? [page.source?.page].filter((value: any) => value !== undefined),
        confidence: normalizeConfidence(record.confidence),
        fields: record.fields ?? Object.fromEntries((page.field_candidates || []).map((field: any) => [
          field.field_name_model || field.label_original || field.field_id,
          field.value_original,
        ])),
        needs_review: record.needs_review === true,
      })));
      continue;
    }

    records.push({
      record_id: `page_${page.source?.page || records.length + 1}`,
      record_type: page.document_classification?.type || "document",
      source_files: [page.source?.source_file].filter(Boolean),
      source_pages: [page.source?.page].filter((value: any) => value !== undefined),
      confidence: page.document_classification?.confidence || "low",
      fields: Object.fromEntries((page.field_candidates || []).map((field: any) => [
        field.field_name_model || field.label_original || field.field_id,
        field.value_original,
      ])),
      needs_review: (page.review_issues || []).some((issue: any) => issue.severity !== "low"),
    });
  }
  return records;
}

function buildFinalJson(semanticPages: any[]): Record<string, any> {
  const out: Record<string, any> = {};
  for (const page of semanticPages) {
    for (const field of page.field_candidates || []) {
      const key = stringifyCell(field.field_name_model || field.label_original || field.field_id);
      if (!key || out[key] !== undefined) continue;
      out[key] = {
        value: field.value_original,
        normalized_value: field.value_normalized,
        label: field.label_original,
        confidence: field.confidence,
        source_page: page.source?.page,
        evidence: field.evidence,
        needs_review: field.needs_review === true,
      };
    }
  }
  return out;
}

function buildAttentionMappedFields(matches: any[]): Record<string, any> {
  const mapped: Record<string, any> = {};
  for (const match of matches) {
    const candidates = Array.isArray(match.candidates) ? match.candidates : [];
    const first = candidates[0] || {};
    mapped[match.name] = {
      value: stringifyCell(first.value_original ?? first.value ?? ""),
      label: stringifyCell(first.label_original ?? first.label ?? ""),
      confidence: normalizeConfidence(first.confidence ?? (match.matched ? "medium" : "low")),
      verified: match.matched === true && match.status === "matched",
      needs_review: match.needs_review === true || match.matched !== true,
      candidates,
      source: "attention_hint_match",
    };
  }
  return mapped;
}

function fieldToLosslessCandidate(field: any): LosslessFieldCandidate {
  return {
    name: stringifyCell(field.field_name_model || field.name || ""),
    label: stringifyCell(field.label_original || field.label || ""),
    value: stringifyCell(field.value_original || field.value || ""),
    bbox: normalizeBbox(field.bbox ?? field.evidence?.bbox),
    confidence: normalizeConfidence(field.confidence),
    source: "qwen3_vl_universal_page_understanding",
    needs_review: field.needs_review === true,
  };
}

function genericToLosslessCandidate(item: any): LosslessFieldCandidate {
  return {
    name: stringifyCell(item.field_name_model || item.name || ""),
    label: stringifyCell(item.label_original || item.label || ""),
    value: stringifyCell(item.value_original || item.value || item.text || ""),
    bbox: normalizeBbox(item.bbox ?? item.evidence?.bbox),
    confidence: normalizeConfidence(item.confidence),
    source: "qwen3_vl_universal_page_understanding",
    needs_review: item.needs_review !== false,
  };
}

function normalizeAttentionHints(raw: any[] | undefined): AttentionHint[] {
  const seen = new Set<string>();
  const hints: AttentionHint[] = [];
  for (const item of Array.isArray(raw) ? raw : []) {
    const name = typeof item === "string"
      ? item.trim()
      : stringifyCell(item?.name ?? item?.label ?? item?.field ?? "").trim();
    if (!name || seen.has(name)) continue;
    const aliases = [
      name,
      ...(typeof item === "object" && item
        ? [item.label_pattern, item.labelPattern, ...(Array.isArray(item.aliases) ? item.aliases : [])]
        : []),
    ].map(stringifyCell).filter(Boolean);
    seen.add(name);
    hints.push({
      name,
      aliases: [...new Set(aliases)],
      required: typeof item === "object" && item?.required === true,
      original: item,
    });
  }
  return hints;
}

function normalizeTextItems(raw: any): any[] {
  return normalizeGenericArray(raw).map((item) => ({
    text: stringifyCell(item.text ?? item.value ?? item.raw ?? item),
    bbox: normalizeBbox(item.bbox),
    confidence: normalizeConfidence(item.confidence),
    source: stringifyCell(item.source ?? "qwen3_vl_universal_page_understanding"),
  })).filter((item) => item.text);
}

function normalizeTables(raw: any): any[] {
  return normalizeGenericArray(raw).map((table, index) => ({
    table_id: stringifyCell(table.table_id ?? table.id ?? `table_${index + 1}`),
    bbox: normalizeBbox(table.bbox),
    rows: Array.isArray(table.rows) ? table.rows.map((row: any) => Array.isArray(row) ? row.map(stringifyCell) : [stringifyCell(row)]) : [],
    source: stringifyCell(table.source ?? "qwen3_vl_universal_page_understanding"),
    confidence: normalizeConfidence(table.confidence),
  }));
}

function normalizeEvidence(raw: any): any {
  if (!raw || typeof raw !== "object") return raw ? { raw_text: stringifyCell(raw) } : {};
  return {
    ...raw,
    bbox: normalizeBbox(raw.bbox),
  };
}

function hasEvidence(field: any): boolean {
  const evidence = field.evidence;
  if (!evidence || typeof evidence !== "object") return false;
  return Boolean(
    stringifyCell(evidence.raw_text || evidence.text || evidence.nearby_text || evidence.table_id || evidence.row_index) ||
    evidence.bbox ||
    field.bbox,
  );
}

function normalizeGenericArray(raw: any): any[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (typeof raw === "object") return Object.values(raw);
  return [raw];
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

function normalizeConfidence(value: any): Confidence {
  const text = String(value || "").toLowerCase();
  if (text === "high" || text === "medium" || text === "low") return text;
  return "low";
}

function reindex(items: any[], prefix: string): any[] {
  return items.map((item, index) => ({
    ...item,
    id: item.id || item.entity_id || item.relationship_id || `${prefix}_${String(index + 1).padStart(5, "0")}`,
  }));
}

function stringifyCell(value: any): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value.replace(/\s+/g, " ").trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function parseUniversalModelJson(text: string): any | null {
  const cleaned = String(text || "")
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .trim();
  const sanitized = sanitizeCommonModelJson(cleaned);
  try {
    return JSON.parse(sanitized);
  } catch {
    try {
      return JSON.parse(jsonrepair(sanitized));
    } catch {
      // Continue to brace extraction below.
    }
    const start = sanitized.indexOf("{");
    const end = sanitized.lastIndexOf("}");
    if (start >= 0 && end > start) {
      const sliced = sanitized.slice(start, end + 1);
      try {
        return JSON.parse(sliced);
      } catch {
        try {
          return JSON.parse(jsonrepair(sliced));
        } catch {
          return null;
        }
      }
    }
    return null;
  }
}

function sanitizeCommonModelJson(text: string): string {
  return text.replace(
    /"bbox"\s*:\s*\{\s*"x1"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\}/g,
    (_match, x1, y1, x2, y2) => `"bbox":{"x1":${x1},"y1":${y1},"x2":${x2},"y2":${y2}}`,
  );
}

function emptyLosslessPage(page: number, error: string): LosslessPage {
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
    uncertain_tokens: [{ text: "", context: error }],
    review_required: true,
  };
}

function formatDateTime(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return [
    date.getFullYear(),
    "-",
    pad(date.getMonth() + 1),
    "-",
    pad(date.getDate()),
    " ",
    pad(date.getHours()),
    ":",
    pad(date.getMinutes()),
    ":",
    pad(date.getSeconds()),
  ].join("");
}
