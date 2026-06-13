import { basename } from "path";
import { MODEL } from "../config/constants.js";
import type { LosslessDocumentResult, LosslessFieldCandidate, LosslessPage } from "../config/types.js";

type Confidence = "high" | "medium" | "low";

export interface SemanticOptions {
  sourcePath?: string;
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
}

interface NormalizedAttentionField {
  name: string;
  aliases: string[];
  required: boolean;
  original: any;
}

interface SemanticPageParts {
  rawPage: any;
  semanticPage: any;
}

export function buildUniversalSemanticResult(
  result: LosslessDocumentResult,
  options: SemanticOptions = {}
): LosslessDocumentResult {
  const attentionFields = normalizeAttentionFields(options.attentionFields);
  const parts = result.pages.map((page) => buildSemanticPage(page, result, options, attentionFields));
  const rawPages = parts.map((part) => part.rawPage);
  const semanticPages = parts.map((part) => part.semanticPage);
  const entities = reindexEntities(semanticPages.flatMap((page) => page.entities || []));
  const relationships = reindexRelationships(semanticPages.flatMap((page) => page.relationships || []), entities);
  const detectedDocuments = buildDetectedDocuments(semanticPages);
  const integratedRecords = buildIntegratedRecords(semanticPages, options);
  const reviewIssues = buildReviewIssues(semanticPages, attentionFields);

  return {
    ...result,
    universal_schema: "universal_document_semantics_v1",
    extraction_policy: {
      preserve_all: true,
      extract_all_fields: options.extractAllFields !== false,
      attention_fields_are_hints_only: true,
      domain_hint: options.domainHint || "auto",
      semantic_mode: options.semanticMode || "auto",
      output_grain: options.outputGrain || "auto",
      integration_mode: options.integrationMode || "none",
      model: MODEL,
      max_api_concurrency: options.maxApiConcurrency,
      render_concurrency: options.renderConcurrency,
      writer_mode: options.writerMode,
    },
    raw_pages: rawPages,
    semantic_pages: semanticPages,
    detected_documents: detectedDocuments,
    entities,
    relationships,
    integrated_records: integratedRecords,
    review_issues: reviewIssues,
  };
}

function buildSemanticPage(
  page: LosslessPage,
  result: LosslessDocumentResult,
  options: SemanticOptions,
  attentionFields: NormalizedAttentionField[]
): SemanticPageParts {
  const source = sourceFor(page, result, options);
  const rawText = pageText(page);
  const tables = normalizeTables(page);
  const baseCandidates = normalizeAllFieldCandidates(page, tables, rawText);
  const fieldCandidates = uniqueCandidates(baseCandidates);
  const attentionMatches = buildAttentionMatches(attentionFields, fieldCandidates);
  const orphanValues = buildOrphanValues(rawText, fieldCandidates, source);
  const documentClassification = classifyDocument(rawText, tables, options.domainHint);
  const entities = buildEntities(fieldCandidates, orphanValues, rawText, source);
  const relationships = buildRelationships(fieldCandidates, entities, tables, attentionMatches, source);
  const reviewIssues = buildPageReviewIssues(documentClassification, fieldCandidates, attentionMatches, page, source);

  const rawPage = {
    source,
    raw_content: {
      raw_markdown: page.raw_markdown || "",
      raw_html: page.raw_html || "",
      text_items: page.text_items || [],
      tables,
      uncertain_tokens: page.uncertain_tokens || [],
    },
  };

  const semanticPage = {
    source,
    document_classification: documentClassification,
    raw_content: rawPage.raw_content,
    field_candidates: fieldCandidates,
    attention_field_matches: attentionMatches,
    unmapped_fields: page.unmapped_fields || [],
    orphan_values: orphanValues,
    entities,
    relationships,
    review_issues: reviewIssues,
  };

  return { rawPage, semanticPage };
}

function sourceFor(page: LosslessPage, result: LosslessDocumentResult, options: SemanticOptions): Record<string, any> {
  const sourcePath = options.sourcePath || result.source_path || "";
  return {
    source_path: sourcePath,
    source_file: sourcePath ? basename(sourcePath) : "",
    page: page.page,
    model: MODEL,
    extracted_at: new Date().toISOString(),
    render: {
      render_scale: options.renderScale,
    },
  };
}

function pageText(page: LosslessPage): string {
  const textItems = (page.text_items || []).map((item) => item.text).filter(Boolean).join("\n");
  return [page.raw_markdown || "", page.raw_html || "", textItems].filter(Boolean).join("\n");
}

function normalizeTables(page: LosslessPage): any[] {
  const existing = Array.isArray(page.tables) ? page.tables : [];
  const parsed = parseMarkdownTables(page.raw_markdown || "");
  return [...existing, ...parsed].map((table, index) => ({
    table_id: table.table_id || `table_${index + 1}`,
    bbox: table.bbox,
    rows: Array.isArray(table.rows) ? table.rows : [],
    source: table.source || "lossless_page",
  }));
}

function parseMarkdownTables(raw: string): any[] {
  const rows: string[][] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.includes("&") || !line.includes("\\\\")) continue;
    const cleaned = cleanLatexCell(line);
    if (!cleaned || cleaned.startsWith("\\")) continue;
    const cells = cleaned.split("&").map((cell) => cleanLatexCell(cell)).filter((cell, idx, arr) => cell || arr.length > 1);
    if (cells.some(Boolean)) rows.push(cells);
  }
  return rows.length ? [{ table_id: "parsed_markdown_table_1", rows, source: "raw_markdown_latex" }] : [];
}

function cleanLatexCell(value: any): string {
  let text = String(value ?? "").replace(/```latex|```/g, " ");
  text = text.replace(/\\\\/g, " ");
  text = text.replace(/\\(?:begin|end)\{[^}]+\}/g, " ");
  text = text.replace(/\\(?:hline|cline\{[^}]+\})/g, " ");
  text = text.replace(/\\textbf\{([^}]*)\}/g, "$1");
  text = text.replace(/\\hspace\{[^}]+\}/g, " ");
  text = text.replace(/\\#/g, "#").replace(/\\&/g, "&").replace(/\\\$/g, "$");
  return text.replace(/\s+/g, " ").trim();
}

function normalizeAllFieldCandidates(page: LosslessPage, tables: any[], rawText: string): any[] {
  const candidates: any[] = [];
  for (const candidate of page.field_candidates || []) {
    candidates.push(candidateToSemantic(candidate, "page_field_candidate", page.page));
  }
  for (const [name, entry] of Object.entries(page.mapped_fields || {})) {
    const value = normalizeValue((entry as any)?.value);
    if (!value) continue;
    candidates.push({
      name,
      label: (entry as any)?.label || name,
      value,
      confidence: normalizeConfidence((entry as any)?.confidence),
      source: "mapped_field",
      evidence: { page: page.page, candidates: (entry as any)?.candidates || [] },
      needs_review: (entry as any)?.needs_review === true,
    });
  }
  candidates.push(...candidatesFromTables(tables, page.page));
  candidates.push(...candidatesFromLabeledText(rawText, page.page));
  return candidates.filter((candidate) => candidate.value);
}

function candidateToSemantic(candidate: LosslessFieldCandidate, source: string, page: number): any {
  return {
    name: candidate.name || canonicalName(candidate.label || ""),
    label: candidate.label || candidate.name || "",
    value: normalizeValue(candidate.value),
    bbox: candidate.bbox,
    confidence: normalizeConfidence(candidate.confidence),
    source: candidate.source || source,
    evidence: { page, bbox: candidate.bbox, text: candidate.value },
    needs_review: candidate.needs_review === true || normalizeConfidence(candidate.confidence) === "low",
  };
}

function candidatesFromTables(tables: any[], page: number): any[] {
  const candidates: any[] = [];
  for (const table of tables) {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    rows.forEach((row: any[], rowIndex: number) => {
      const cells = row.map((cell) => cleanLatexCell(cell)).filter((cell) => cell !== "");
      if (cells.length < 2) return;
      if (looksLikeLabel(cells[0]) && cells.slice(1).some((cell) => !looksLikeLabel(cell))) {
        const label = cells[0];
        const value = cells.slice(1).join(" | ");
        candidates.push({
          name: canonicalName(label),
          label,
          value,
          confidence: "medium",
          source: "table_label_value",
          evidence: { page, table_id: table.table_id, row_index: rowIndex + 1, row },
          needs_review: false,
        });
      }
      cells.forEach((cell, colIndex) => {
        const typed = classifyValue(cell);
        if (!typed) return;
        candidates.push({
          name: typed.type,
          label: "",
          value: cell,
          confidence: "low",
          source: "typed_table_cell",
          evidence: { page, table_id: table.table_id, row_index: rowIndex + 1, column_index: colIndex + 1, row },
          needs_review: true,
        });
      });
    });
  }
  return candidates;
}

function candidatesFromLabeledText(rawText: string, page: number): any[] {
  const candidates: any[] = [];
  const lines = rawText.split(/\r?\n/).map((line) => cleanLatexCell(line)).filter(Boolean);
  for (const line of lines) {
    const colon = line.match(/^(.{2,60}?)[：:]\s*(.{1,160})$/);
    if (colon && looksLikeLabel(colon[1])) {
      candidates.push({
        name: canonicalName(colon[1]),
        label: colon[1].trim(),
        value: colon[2].trim(),
        confidence: "medium",
        source: "labeled_text",
        evidence: { page, text: line },
        needs_review: false,
      });
      continue;
    }
    const pair = line.match(/^([A-Za-z][A-Za-z0-9 /().#-]{2,40})\s{2,}(.{2,120})$/);
    if (pair && looksLikeLabel(pair[1])) {
      candidates.push({
        name: canonicalName(pair[1]),
        label: pair[1].trim(),
        value: pair[2].trim(),
        confidence: "low",
        source: "spaced_labeled_text",
        evidence: { page, text: line },
        needs_review: true,
      });
    }
  }
  return candidates;
}

function normalizeAttentionFields(raw: any[] | undefined): NormalizedAttentionField[] {
  return (Array.isArray(raw) ? raw : []).map((item) => {
    if (typeof item === "string") {
      return { name: item.trim(), aliases: [item.trim()], required: false, original: item };
    }
    const name = String(item?.name || item?.label || item?.field || "").trim();
    const aliases = [
      name,
      item?.label_pattern,
      item?.labelPattern,
      ...(Array.isArray(item?.aliases) ? item.aliases : []),
    ].map((v) => String(v || "").trim()).filter(Boolean);
    return { name, aliases: [...new Set(aliases)], required: item?.required === true, original: item };
  }).filter((item) => item.name);
}

function buildAttentionMatches(attentionFields: NormalizedAttentionField[], candidates: any[]): any[] {
  return attentionFields.map((field) => {
    const aliases = field.aliases.length ? field.aliases : [field.name];
    const matches = candidates.filter((candidate) =>
      aliases.some((alias) => candidateMatchesAttention(candidate, alias))
    );
    return {
      name: field.name,
      aliases,
      matched: matches.length > 0,
      required: field.required,
      candidates: matches,
      needs_review: field.required && matches.length === 0,
    };
  });
}

function candidateMatchesAttention(candidate: any, alias: string): boolean {
  const hay = normalizeComparable([candidate.name, candidate.label, candidate.value].filter(Boolean).join(" "));
  const needle = normalizeComparable(alias);
  return !!needle && (hay.includes(needle) || needle.includes(hay));
}

function buildOrphanValues(rawText: string, candidates: any[], source: any): any[] {
  const known = new Set(candidates.map((candidate) => normalizeComparable(candidate.value)));
  const found: any[] = [];
  for (const match of findTypedValues(rawText)) {
    if (known.has(normalizeComparable(match.value))) continue;
    found.push({ ...match, confidence: "low", source: { ...source, extraction: "regex_orphan_value" }, needs_review: true });
  }
  return uniqueBy(found, (item) => `${item.type}:${normalizeComparable(item.value)}`).slice(0, 200);
}

function classifyDocument(rawText: string, tables: any[], domainHint?: string): any {
  const text = normalizeComparable(rawText);
  const scores: Record<string, number> = {
    bank_statement: score(text, ["statement", "account", "balance", "deposit", "withdrawal", "cheque", "transaction", "bank"]),
    invoice: score(text, ["invoice", "amount", "total", "client", "customer", "date"]),
    shipping_document: score(text, ["vessel", "voyage", "container", "billoflading", "port", "receipt", "shipper", "consignee"]),
    reimbursement_record: score(text, ["reimbursement", "voucher", "grossamount", "netinvoice", "paydiscount", "transactionadvice"]),
    calculation_sheet: score(text, ["subtotal", "provision", "batch", "count", "worksheet", "file no", "fileno"]),
    table_document: tables.length ? 2 : 0,
  };
  const hint = normalizeComparable(domainHint || "");
  if (hint && hint !== "auto") {
    for (const key of Object.keys(scores)) {
      if (normalizeComparable(key).includes(hint) || hint.includes(normalizeComparable(key))) scores[key] += 2;
    }
  }
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [type, bestScore] = sorted[0] || ["document", 0];
  return {
    type: bestScore > 0 ? type : "document",
    confidence: bestScore >= 4 ? "high" : bestScore >= 2 ? "medium" : "low",
    reasons: sorted.filter(([, value]) => value > 0).slice(0, 4).map(([name, value]) => ({ name, score: value })),
  };
}

function buildEntities(candidates: any[], orphanValues: any[], rawText: string, source: any): any[] {
  const entities: any[] = [];
  for (const candidate of candidates) {
    const typed = classifyValue(candidate.value) || classifyFieldName(candidate.name || candidate.label || "");
    if (!typed) continue;
    entities.push({
      id: "",
      type: typed.type,
      value: candidate.value,
      label: candidate.label || candidate.name || "",
      confidence: candidate.confidence || typed.confidence,
      source: { ...source, evidence: candidate.evidence },
      needs_review: candidate.needs_review === true,
    });
  }
  for (const orphan of orphanValues) {
    entities.push({
      id: "",
      type: orphan.type,
      value: orphan.value,
      label: "",
      confidence: orphan.confidence || "low",
      source: orphan.source || source,
      needs_review: true,
    });
  }
  for (const organization of findOrganizations(rawText).slice(0, 50)) {
    entities.push({
      id: "",
      type: "organization",
      value: organization,
      label: "",
      confidence: "medium",
      source,
      needs_review: false,
    });
  }
  return uniqueBy(entities, (item) => `${item.type}:${normalizeComparable(item.value)}`);
}

function buildRelationships(candidates: any[], entities: any[], tables: any[], attentionMatches: any[], source: any): any[] {
  const relationships: any[] = [];
  for (const candidate of candidates) {
    const matched = entities.find((entity) => normalizeComparable(entity.value) === normalizeComparable(candidate.value));
    if (!matched) continue;
    relationships.push({
      id: "",
      type: "field_mentions_entity",
      from: candidate.name || candidate.label || "field",
      to: matched.id || `${matched.type}:${matched.value}`,
      confidence: candidate.confidence || "medium",
      evidence: candidate.evidence,
      source,
    });
  }
  for (const table of tables) {
    (table.rows || []).forEach((row: any[], index: number) => {
      relationships.push({
        id: "",
        type: "table_row_groups_values",
        from: table.table_id,
        to: `row_${index + 1}`,
        confidence: "medium",
        evidence: { row },
        source,
      });
    });
  }
  for (const match of attentionMatches.filter((item) => item.matched)) {
    relationships.push({
      id: "",
      type: "attention_field_matched",
      from: match.name,
      to: match.candidates.map((candidate: any) => candidate.name || candidate.label || candidate.value),
      confidence: match.candidates.some((candidate: any) => candidate.confidence === "high") ? "high" : "medium",
      evidence: match.candidates.map((candidate: any) => candidate.evidence),
      source,
    });
  }
  return relationships;
}

function buildDetectedDocuments(semanticPages: any[]): any[] {
  const groups = new Map<string, any>();
  for (const page of semanticPages) {
    const cls = page.document_classification || { type: "document", confidence: "low" };
    const key = `${page.source.source_file}:${cls.type}`;
    const group = groups.get(key) || {
      source_file: page.source.source_file,
      document_type: cls.type,
      confidence: cls.confidence,
      pages: [],
      reasons: [],
    };
    group.pages.push(page.source.page);
    group.reasons.push(...(cls.reasons || []));
    groups.set(key, group);
  }
  return [...groups.values()].map((group) => ({
    ...group,
    pages: ([...new Set<number>(group.pages as number[])]).sort((a, b) => a - b),
    reasons: group.reasons.slice(0, 20),
  }));
}

function buildIntegratedRecords(semanticPages: any[], options: SemanticOptions): any[] {
  const records: any[] = [];
  for (const page of semanticPages) {
    if (options.outputGrain === "row" || options.outputGrain === "transaction") {
      const rowRecords = recordsFromTables(page);
      if (rowRecords.length) {
        records.push(...rowRecords);
        continue;
      }
    }
    const record: Record<string, any> = {
      record_id: `page_${page.source.page}`,
      record_type: page.document_classification?.type || "document",
      confidence: page.document_classification?.confidence || "low",
      source_files: [page.source.source_file].filter(Boolean),
      source_pages: [page.source.page],
    };
    for (const candidate of page.field_candidates || []) {
      const key = candidate.name || canonicalName(candidate.label || "field");
      if (!key || record[key] !== undefined) continue;
      record[key] = candidate.value;
    }
    records.push(record);
  }
  return records;
}

function recordsFromTables(page: any): any[] {
  const records: any[] = [];
  for (const table of page.raw_content?.tables || []) {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    if (rows.length < 2) continue;
    const header = rows[0].map((cell: any, idx: number) => canonicalName(cleanLatexCell(cell)) || `col_${idx + 1}`);
    rows.slice(1).forEach((row: any[], index: number) => {
      const record: Record<string, any> = {
        record_id: `${table.table_id}_row_${index + 1}`,
        record_type: page.document_classification?.type || "table_row",
        source_files: [page.source.source_file].filter(Boolean),
        source_pages: [page.source.page],
      };
      row.forEach((cell, colIndex) => {
        record[header[colIndex] || `col_${colIndex + 1}`] = cleanLatexCell(cell);
      });
      records.push(record);
    });
  }
  return records;
}

function buildReviewIssues(semanticPages: any[], attentionFields: NormalizedAttentionField[]): any[] {
  return semanticPages.flatMap((page) => page.review_issues || []).concat(
    attentionFields.flatMap((field) => {
      const anyMatch = semanticPages.some((page) =>
        (page.attention_field_matches || []).some((match: any) => match.name === field.name && match.matched)
      );
      return anyMatch ? [] : [{
        severity: field.required ? "high" : "medium",
        type: "attention_field_not_found",
        message: `Attention field not found: ${field.name}`,
        field: field.name,
      }];
    })
  );
}

function buildPageReviewIssues(
  classification: any,
  candidates: any[],
  attentionMatches: any[],
  page: LosslessPage,
  source: any
): any[] {
  const issues: any[] = [];
  if (classification.confidence === "low") {
    issues.push({ severity: "medium", type: "low_classification_confidence", source, message: "Document type classification is uncertain." });
  }
  if (!candidates.length) {
    issues.push({ severity: "high", type: "no_field_candidates", source, message: "No structured field candidates were detected." });
  }
  for (const match of attentionMatches.filter((item) => item.required && !item.matched)) {
    issues.push({ severity: "high", type: "required_attention_field_missing", source, field: match.name, message: `Required attention field missing: ${match.name}` });
  }
  if (page.review_required) {
    issues.push({ severity: "medium", type: "page_review_required", source, message: "Lossless extraction marked this page as review required." });
  }
  return issues;
}

function reindexEntities(entities: any[]): any[] {
  return uniqueBy(entities, (item) => `${item.type}:${normalizeComparable(item.value)}`)
    .map((entity, index) => ({ ...entity, id: entity.id || `ent_${String(index + 1).padStart(5, "0")}` }));
}

function reindexRelationships(relationships: any[], entities: any[]): any[] {
  const entityByValue = new Map(entities.map((entity) => [`${entity.type}:${normalizeComparable(entity.value)}`, entity.id]));
  return relationships.map((rel, index) => {
    const to = typeof rel.to === "string" ? entityByValue.get(rel.to) || rel.to : rel.to;
    return { ...rel, id: rel.id || `rel_${String(index + 1).padStart(5, "0")}`, to };
  });
}

function findTypedValues(text: string): any[] {
  const values: any[] = [];
  const patterns: Array<[string, RegExp]> = [
    ["amount", /\b(?:HKD|USD|RMB|CNY|\$)\s*[-+]?\d[\d,]*(?:\.\d{1,2})?\b/gi],
    ["date", /\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b/g],
    ["identifier", /\b[A-Z]{2,6}\d{4,12}[A-Z0-9-]*\b/g],
    ["account_number", /\b\d{3}-\d{6}-\d{3}\b/g],
    ["code", /\b[0-9A-Z]{1,4}[A-Z]-[0-9A-Z]{1,6}\b/g],
  ];
  for (const [type, regex] of patterns) {
    for (const match of text.matchAll(regex)) {
      values.push({ type, value: match[0] });
    }
  }
  return values;
}

function findOrganizations(text: string): string[] {
  const organizations: string[] = [];
  const regex = /\b[A-Z][A-Z0-9 &'().,-]{2,80}\b(?:LIMITED|LTD\.?|COMPANY|BANK|CORPORATION|INC\.?)\b/gi;
  for (const match of text.matchAll(regex)) {
    organizations.push(match[0].replace(/\s+/g, " ").trim());
  }
  return [...new Set(organizations)];
}

function classifyValue(value: string): { type: string; confidence: Confidence } | null {
  const text = String(value || "").trim();
  if (!text) return null;
  if (/^(?:HKD|USD|RMB|CNY|\$)?\s*[-+]?\d[\d,]*(?:\.\d{1,2})?$/.test(text)) return { type: "amount", confidence: "medium" };
  if (/^(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})$/.test(text)) return { type: "date", confidence: "medium" };
  if (/^[A-Z]{4}\d{7}$/.test(text)) return { type: "container_number", confidence: "high" };
  if (/^\d{3}-\d{6}-\d{3}$/.test(text)) return { type: "account_number", confidence: "high" };
  if (/^[A-Z]{2,6}\d{4,12}[A-Z0-9-]*$/.test(text)) return { type: "identifier", confidence: "medium" };
  return null;
}

function classifyFieldName(name: string): { type: string; confidence: Confidence } | null {
  const normalized = normalizeComparable(name);
  if (!normalized) return null;
  if (normalized.includes("amount") || normalized.includes("total") || normalized.includes("金額")) return { type: "amount", confidence: "medium" };
  if (normalized.includes("date") || normalized.includes("日期")) return { type: "date", confidence: "medium" };
  if (normalized.includes("cheque") || normalized.includes("check") || normalized.includes("支票")) return { type: "cheque", confidence: "medium" };
  if (normalized.includes("invoice") || normalized.includes("發票")) return { type: "invoice", confidence: "medium" };
  if (normalized.includes("account") || normalized.includes("戶口")) return { type: "account", confidence: "medium" };
  if (normalized.includes("company") || normalized.includes("vendor") || normalized.includes("client") || normalized.includes("公司")) return { type: "organization", confidence: "medium" };
  return null;
}

function score(text: string, keywords: string[]): number {
  return keywords.reduce((sum, keyword) => sum + (text.includes(normalizeComparable(keyword)) ? 1 : 0), 0);
}

function looksLikeLabel(value: string): boolean {
  const text = String(value || "").trim();
  if (text.length < 2 || text.length > 80) return false;
  if (classifyValue(text)) return false;
  return /[A-Za-z\u4e00-\u9fff]/.test(text);
}

function canonicalName(label: string): string {
  const normalized = normalizeComparable(label)
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || "";
}

function normalizeValue(value: any): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") {
    if ("value" in value) return normalizeValue((value as any).value);
    if ("normalized_value" in value) return normalizeValue((value as any).normalized_value);
  }
  return String(value).replace(/\s+/g, " ").trim();
}

function normalizeConfidence(value: any): Confidence {
  return value === "high" || value === "medium" || value === "low" ? value : "low";
}

function normalizeComparable(value: string): string {
  return String(value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\s_.,:;'"()[\]{}|\\/-]+/g, "");
}

function uniqueCandidates(candidates: any[]): any[] {
  return uniqueBy(candidates, (candidate) =>
    `${candidate.name || ""}:${candidate.label || ""}:${normalizeComparable(candidate.value || "")}`
  );
}

function uniqueBy<T>(items: T[], keyFn: (item: T) => string): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const item of items) {
    const key = keyFn(item);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}
