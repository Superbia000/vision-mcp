import { appendFileSync, existsSync, mkdirSync, renameSync, writeFileSync } from "fs";
import { dirname, join, parse } from "path";
import ExcelJS from "exceljs";
import type { LosslessDocumentResult } from "../config/types.js";

export interface UniversalWriterOptions {
  outputDir: string;
  sourcePath?: string;
  writerMode?: string;
  exportFormats?: string[];
}

export function writeUniversalPageCheckpoint(
  result: LosslessDocumentResult,
  options: UniversalWriterOptions,
): any[] {
  if (!isUniversal(result) || !options.outputDir) return [];
  ensureOutputDirs(options.outputDir);

  const artifacts: any[] = [];
  for (const page of result.semantic_pages || []) {
    const pageNumber = Number(page?.source?.page || 0);
    const pagePath = pageJsonPath(options.outputDir, page?.source?.source_path || options.sourcePath || result.source_path || "", pageNumber);
    if (existsSync(pagePath)) continue;

    const pageResult = {
      schema: result.schema,
      universal_schema: result.universal_schema,
      extraction_policy: result.extraction_policy,
      raw_pages: (result.raw_pages || []).filter((raw) => Number(raw?.source?.page || 0) === pageNumber),
      semantic_pages: [page],
      entities: (page.entities || []),
      relationships: (page.relationships || []),
      integrated_records: (page.integrated_records || []),
      review_issues: (page.review_issues || []),
    };

    writeJsonAtomic(pagePath, pageResult);
    appendNdjson(join(options.outputDir, "raw_pages.ndjson"), pageResult.raw_pages || []);
    appendNdjson(join(options.outputDir, "semantic_pages.ndjson"), pageResult.semantic_pages || []);
    appendNdjson(join(options.outputDir, "entities.ndjson"), page.entities || []);
    appendNdjson(join(options.outputDir, "relationships.ndjson"), page.relationships || []);
    appendNdjson(join(options.outputDir, "review_issues.ndjson"), page.review_issues || []);
    appendNdjson(join(options.outputDir, "raw_page_index.ndjson"), [{
      source_pdf: page?.source?.source_path || options.sourcePath || result.source_path || "",
      source_file: page?.source?.source_file || "",
      page_number: pageNumber,
      page_json_path: pagePath,
      extracted_at: page?.source?.extracted_at,
      model: page?.source?.model,
    }]);

    artifacts.push({
      role: "universal_page_checkpoint",
      page: pageNumber,
      path: pagePath,
      format: "json",
    });
  }

  return artifacts;
}

export async function writeUniversalFinalOutputs(
  result: LosslessDocumentResult,
  options: UniversalWriterOptions,
): Promise<any[]> {
  if (!isUniversal(result) || !options.outputDir || options.writerMode === "none") return [];
  ensureOutputDirs(options.outputDir);

  const formats = normalizeFormats(options.exportFormats);
  const artifacts: any[] = [];
  const fullJsonPath = join(options.outputDir, "extraction_result.json");
  const recordsPath = join(options.outputDir, "integrated_records.json");
  writeJsonAtomic(fullJsonPath, result);
  writeJsonAtomic(recordsPath, result.integrated_records || []);
  artifacts.push({ role: "universal_full_json", path: fullJsonPath, format: "json" });
  artifacts.push({ role: "universal_integrated_records", path: recordsPath, format: "json" });

  ensureNdjsonExists(join(options.outputDir, "raw_pages.ndjson"), result.raw_pages || []);
  ensureNdjsonExists(join(options.outputDir, "semantic_pages.ndjson"), result.semantic_pages || []);
  ensureNdjsonExists(join(options.outputDir, "entities.ndjson"), result.entities || []);
  ensureNdjsonExists(join(options.outputDir, "relationships.ndjson"), result.relationships || []);
  ensureNdjsonExists(join(options.outputDir, "review_issues.ndjson"), result.review_issues || []);

  if (formats.includes("markdown")) {
    const markdownPath = join(options.outputDir, "report.md");
    writeFileAtomic(markdownPath, buildMarkdownReport(result));
    artifacts.push({ role: "universal_markdown_report", path: markdownPath, format: "markdown" });
  }

  if (formats.includes("xlsx")) {
    const xlsxPath = join(options.outputDir, "extraction.xlsx");
    await writeWorkbookAtomic(xlsxPath, result);
    artifacts.push({ role: "universal_excel_workbook", path: xlsxPath, format: "xlsx" });
  }

  return artifacts;
}

async function writeWorkbookAtomic(path: string, result: LosslessDocumentResult): Promise<void> {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "vision-mcp";
  workbook.created = new Date();
  addSheet(workbook, "Overview", overviewRows(result));
  addSheet(workbook, "Integrated Records", integratedRecordRows(result));
  addSheet(workbook, "Detected Fields", detectedFieldRows(result));
  addSheet(workbook, "Tables", tableRows(result));
  addSheet(workbook, "Entities", (result.entities || []).map(flattenRow));
  addSheet(workbook, "Relationships", (result.relationships || []).map(flattenRow));
  addSheet(workbook, "Review Issues", (result.review_issues || []).map(flattenRow));
  addSheet(workbook, "Raw Page Index", rawPageIndexRows(result));
  const attentionRows = attentionFieldRows(result);
  if (attentionRows.length) addSheet(workbook, "Attention Fields", attentionRows);

  const tmp = `${path}.tmp`;
  await workbook.xlsx.writeFile(tmp);
  renameSync(tmp, path);
}

function addSheet(workbook: ExcelJS.Workbook, name: string, rows: Record<string, any>[]): void {
  const safeRows = rows.length ? rows : [{ status: "empty" }];
  const worksheet = workbook.addWorksheet(sanitizeSheetName(name));
  const keys = Object.keys(safeRows[0] || {});
  const widths = inferColumnWidths(safeRows, keys);
  worksheet.columns = keys.map((key, index) => ({
    header: key,
    key,
    width: widths[index],
  }));
  for (const row of safeRows) worksheet.addRow(row);
  worksheet.getRow(1).font = { bold: true };
  worksheet.getRow(1).alignment = { vertical: "middle", wrapText: true };
  worksheet.views = [{ state: "frozen", ySplit: 1 }];
}

function overviewRows(result: LosslessDocumentResult): Record<string, any>[] {
  const detected = result.detected_documents || [];
  if (!detected.length) {
    return [{
      schema: result.universal_schema || "",
      source_path: result.source_path || "",
      page_count: result.pages?.length || 0,
      review_required: result.review_required,
      review_issue_count: result.review_issues?.length || 0,
      model: result.extraction_policy?.model || "",
      generated_at: result.extraction_policy?.generated_at || "",
    }];
  }
  return detected.map((doc) => ({
    schema: result.universal_schema || "",
    source_path: result.source_path || "",
    source_file: doc.source_file || "",
    document_id: doc.document_id || "",
    document_type: doc.document_type || "",
    confidence: doc.confidence || "",
    pages: Array.isArray(doc.pages) ? doc.pages.join(",") : stringify(doc.pages),
    reason: doc.reason || "",
    needs_review: doc.needs_review === true,
    total_pages: result.pages?.length || 0,
    review_issue_count: result.review_issues?.length || 0,
    model: result.extraction_policy?.model || "",
    generated_at: result.extraction_policy?.generated_at || "",
  }));
}

function integratedRecordRows(result: LosslessDocumentResult): Record<string, any>[] {
  return (result.integrated_records || []).map((record) => {
    const fields = record.fields && typeof record.fields === "object" ? record.fields : {};
    return flattenRow({
      record_id: record.record_id,
      record_type: record.record_type,
      confidence: record.confidence,
      source_files: Array.isArray(record.source_files) ? record.source_files.join("; ") : record.source_files,
      source_pages: Array.isArray(record.source_pages) ? record.source_pages.join(",") : record.source_pages,
      needs_review: record.needs_review === true,
      ...fields,
    });
  });
}

function detectedFieldRows(result: LosslessDocumentResult): Record<string, any>[] {
  const rows: Record<string, any>[] = [];
  for (const page of result.semantic_pages || []) {
    for (const field of page.field_candidates || []) {
      rows.push(flattenRow({
        source_pdf: field.source_pdf || page.source?.source_path || result.source_path || "",
        source_file: page.source?.source_file || "",
        page_number: field.page_number || page.source?.page,
        document_type: page.document_classification?.type || "",
        label_original: field.label_original || field.label || "",
        field_name_model: field.field_name_model || field.name || "",
        value_original: field.value_original || field.value || "",
        value_normalized: field.value_normalized || "",
        value_type_model: field.value_type_model || "",
        confidence: field.confidence || "",
        needs_review: field.needs_review === true,
        attention_match: stringify(field.attention_match),
        evidence: stringify(field.evidence),
      }));
    }
  }
  return rows;
}

function tableRows(result: LosslessDocumentResult): Record<string, any>[] {
  const rows: Record<string, any>[] = [];
  for (const page of result.semantic_pages || []) {
    for (const table of page.raw_content?.tables || []) {
      const tableRows = Array.isArray(table.rows) ? table.rows : [];
      tableRows.forEach((row: any[], rowIndex: number) => {
        const output: Record<string, any> = {
          source_pdf: page.source?.source_path || result.source_path || "",
          source_file: page.source?.source_file || "",
          page_number: page.source?.page,
          document_type: page.document_classification?.type || "",
          table_id: table.table_id || "",
          row_index: rowIndex + 1,
        };
        row.forEach((cell, index) => {
          output[`col_${index + 1}`] = stringify(cell);
        });
        rows.push(output);
      });
    }
  }
  return rows;
}

function rawPageIndexRows(result: LosslessDocumentResult): Record<string, any>[] {
  return (result.raw_pages || []).map((page) => ({
    source_pdf: page.source?.source_path || result.source_path || "",
    source_file: page.source?.source_file || "",
    page_number: page.source?.page,
    model: page.source?.model || result.extraction_policy?.model || "",
    extracted_at: page.source?.extracted_at || "",
    raw_markdown_chars: String(page.raw_content?.raw_markdown || "").length,
    text_item_count: page.raw_content?.text_items?.length || 0,
    table_count: page.raw_content?.tables?.length || 0,
  }));
}

function attentionFieldRows(result: LosslessDocumentResult): Record<string, any>[] {
  const rows: Record<string, any>[] = [];
  for (const page of result.semantic_pages || []) {
    for (const match of page.attention_field_matches || []) {
      rows.push(flattenRow({
        source_pdf: page.source?.source_path || result.source_path || "",
        source_file: page.source?.source_file || "",
        page_number: page.source?.page,
        name: match.name,
        aliases: Array.isArray(match.aliases) ? match.aliases.join("; ") : stringify(match.aliases),
        status: match.status,
        matched: match.matched === true,
        needs_review: match.needs_review === true,
        candidate_count: Array.isArray(match.candidates) ? match.candidates.length : 0,
        candidates: stringify(match.candidates),
        reason: match.reason || "",
      }));
    }
  }
  return rows;
}

function buildMarkdownReport(result: LosslessDocumentResult): string {
  const lines: string[] = [];
  lines.push("# Universal Document Extraction Report");
  lines.push("");
  lines.push(`- Schema: ${result.universal_schema || ""}`);
  lines.push(`- Source: ${result.source_path || ""}`);
  lines.push(`- Pages: ${result.pages?.length || 0}`);
  lines.push(`- Model: ${result.extraction_policy?.model || ""}`);
  lines.push(`- Generated: ${result.extraction_policy?.generated_at || ""}`);
  lines.push(`- Review required: ${result.review_required ? "yes" : "no"}`);
  lines.push("");

  lines.push("## Document Classification");
  for (const doc of result.detected_documents || []) {
    lines.push(`- ${doc.source_file || ""} pages ${Array.isArray(doc.pages) ? doc.pages.join(",") : ""}: ${doc.document_type || "document"} (${doc.confidence || "low"})`);
  }
  lines.push("");

  lines.push("## Integrated Records");
  const records = result.integrated_records || [];
  if (!records.length) {
    lines.push("No integrated records returned.");
  } else {
    for (const record of records.slice(0, 100)) {
      lines.push(`- ${record.record_id || ""} ${record.record_type || ""} pages ${Array.isArray(record.source_pages) ? record.source_pages.join(",") : ""}`);
    }
    if (records.length > 100) lines.push(`- ... ${records.length - 100} more records`);
  }
  lines.push("");

  lines.push("## Review Issues");
  const issues = result.review_issues || [];
  if (!issues.length) {
    lines.push("No review issues returned.");
  } else {
    for (const issue of issues.slice(0, 200)) {
      lines.push(`- [${issue.severity || "medium"}] ${issue.type || "issue"} page ${issue.source?.page ?? ""}: ${issue.message || ""}`);
    }
    if (issues.length > 200) lines.push(`- ... ${issues.length - 200} more issues`);
  }
  lines.push("");

  lines.push("## Raw Data");
  lines.push("- Raw OCR/layout content is stored in `raw_pages.ndjson` and per-page JSON files.");
  lines.push("- Structured page semantics are stored in `semantic_pages.ndjson`.");
  return `${lines.join("\n")}\n`;
}

function flattenRow(row: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const [key, value] of Object.entries(row || {})) {
    out[key] = stringify(value);
  }
  return out;
}

function appendNdjson(path: string, rows: any[]): void {
  if (!rows.length) return;
  appendFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n") + "\n", "utf8");
}

function ensureNdjsonExists(path: string, rows: any[]): void {
  if (existsSync(path)) return;
  writeFileAtomic(path, rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length ? "\n" : ""));
}

function writeJsonAtomic(path: string, value: any): void {
  writeFileAtomic(path, `${JSON.stringify(value, null, 2)}\n`);
}

function writeFileAtomic(path: string, value: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, value, "utf8");
  renameSync(tmp, path);
}

function ensureOutputDirs(outputDir: string): void {
  mkdirSync(outputDir, { recursive: true });
  mkdirSync(join(outputDir, "pages"), { recursive: true });
}

function pageJsonPath(outputDir: string, sourcePath: string, pageNumber: number): string {
  const base = sanitizeFileBase(parse(sourcePath || "document").name || "document");
  return join(outputDir, "pages", `${base}-p${String(pageNumber).padStart(4, "0")}.json`);
}

function normalizeFormats(raw: string[] | undefined): string[] {
  const formats = Array.isArray(raw) && raw.length ? raw.map((item) => String(item).toLowerCase()) : ["json", "jsonl", "xlsx", "markdown"];
  return [...new Set(formats)];
}

function inferColumnWidths(rows: Record<string, any>[], keys: string[]): number[] {
  return keys.map((key) => {
    const max = Math.max(
      key.length,
      ...rows.slice(0, 100).map((row) => String(row[key] ?? "").slice(0, 80).length),
    );
    return Math.min(Math.max(max + 2, 10), 60);
  });
}

function sanitizeSheetName(name: string): string {
  return name.replace(/[\\/?*[\]:]/g, " ").slice(0, 31) || "Sheet";
}

function sanitizeFileBase(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1F]/g, "_").slice(0, 80) || "document";
}

function stringify(value: any): any {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isUniversal(result: LosslessDocumentResult): boolean {
  return result?.universal_schema === "universal_document_semantics_v2";
}
