import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import ExcelJS from "exceljs";
import {
  writeUniversalFinalOutputs,
  writeUniversalPageCheckpoint,
} from "../dist/output/universal-writer.js";
import { parseUniversalModelJson } from "../dist/extraction/universal-model.js";

test("universal model prompt source has no default business-field template", () => {
  const source = readFileSync(new URL("../src/extraction/universal-model.ts", import.meta.url), "utf8");
  for (const forbidden of [
    "shipping",
    "bank_statement",
    "container",
    "vessel",
    "voyage",
    "cheque",
    "invoice",
    "receipt",
    "charge_code",
    "qwen-vl-ocr",
  ]) {
    assert.equal(source.toLowerCase().includes(forbidden), false, `unexpected default domain term: ${forbidden}`);
  }
  assert.equal(source.includes("attention fields are hints only"), true);
  assert.equal(source.includes("single visual model universal OCR/KIE"), true);
});

test("writer preserves non-attention fields and creates JSONL, Markdown, and XLSX artifacts", async () => {
  const outputDir = mkdtempSync(join(tmpdir(), "vision-mcp-universal-"));
  const result = sampleUniversalResult();

  const checkpointArtifacts = writeUniversalPageCheckpoint(result, {
    outputDir,
    sourcePath: result.source_path,
    writerMode: "jsonl_checkpoint_then_bulk_export",
  });
  const finalArtifacts = await writeUniversalFinalOutputs(result, {
    outputDir,
    sourcePath: result.source_path,
    writerMode: "jsonl_checkpoint_then_bulk_export",
    exportFormats: ["jsonl", "json", "xlsx", "markdown"],
  });

  assert.equal(checkpointArtifacts.length, 1);
  assert.ok(finalArtifacts.some((artifact) => artifact.format === "xlsx"));
  assert.ok(existsSync(join(outputDir, "raw_pages.ndjson")));
  assert.ok(existsSync(join(outputDir, "semantic_pages.ndjson")));
  assert.ok(existsSync(join(outputDir, "integrated_records.json")));
  assert.ok(existsSync(join(outputDir, "report.md")));
  assert.ok(existsSync(join(outputDir, "extraction.xlsx")));

  const semanticLines = readFileSync(join(outputDir, "semantic_pages.ndjson"), "utf8").trim().split(/\r?\n/);
  assert.equal(semanticLines.length, 1);

  const workbook = new ExcelJS.Workbook();
  await workbook.xlsx.readFile(join(outputDir, "extraction.xlsx"));
  const sheetNames = workbook.worksheets.map((worksheet) => worksheet.name);
  for (const sheetName of [
    "Overview",
    "Integrated Records",
    "Detected Fields",
    "Tables",
    "Entities",
    "Relationships",
    "Review Issues",
    "Raw Page Index",
    "Attention Fields",
  ]) {
    assert.ok(sheetNames.includes(sheetName), `missing sheet: ${sheetName}`);
  }

  const detectedSheet = workbook.getWorksheet("Detected Fields");
  const detectedRows = worksheetRows(detectedSheet);
  assert.equal(detectedRows.some((row) => row.field_name_model === "attention_target"), true);
  assert.equal(detectedRows.some((row) => row.field_name_model === "other_visible_field"), true);
  const formulaCells = [];
  detectedSheet.eachRow((row) => row.eachCell((cell) => {
    if (cell.formula) formulaCells.push(cell.address);
  }));
  assert.equal(formulaCells.length, 0);
});

test("parser repairs common malformed model bbox JSON", () => {
  const parsed = parseUniversalModelJson(`{
    "schema": "universal_document_semantics_v2",
    "field_candidates": [{
      "field_name_model": "visible_field",
      "value_original": "ABC",
      "evidence": {
        "bbox": { "x1": 10, 20, 30, 40 }
      }
    }]
  }`);
  assert.equal(parsed.schema, "universal_document_semantics_v2");
  assert.deepEqual(parsed.field_candidates[0].evidence.bbox, { x1: 10, y1: 20, x2: 30, y2: 40 });
});

function sampleUniversalResult() {
  const source = {
    source_path: "C:/sample/source.pdf",
    source_file: "source.pdf",
    page: 1,
    model: "qwen3-vl-plus",
    extracted_at: "2026-06-14 04:30:00",
    render: { render_scale: 1 },
  };
  const rawContent = {
    raw_markdown: "Visible Label: ABC123\nOther Label: XYZ789\n| A | B |\n| 1 | 2 |",
    raw_html: "",
    text_items: [{ text: "Visible Label: ABC123", confidence: "high" }],
    tables: [{ table_id: "table_1", rows: [["A", "B"], ["1", "2"]] }],
    uncertain_tokens: [],
  };
  const fields = [
    {
      field_id: "field_p1_1",
      source_pdf: source.source_path,
      page_number: 1,
      label_original: "Visible Label",
      field_name_model: "attention_target",
      value_original: "ABC123",
      value_normalized: "ABC123",
      value_type_model: "identifier",
      confidence: "high",
      evidence: { page: 1, raw_text: "Visible Label: ABC123" },
      attention_match: "Attention Target",
      needs_review: false,
    },
    {
      field_id: "field_p1_2",
      source_pdf: source.source_path,
      page_number: 1,
      label_original: "Other Label",
      field_name_model: "other_visible_field",
      value_original: "XYZ789",
      value_normalized: "XYZ789",
      value_type_model: "identifier",
      confidence: "high",
      evidence: { page: 1, raw_text: "Other Label: XYZ789" },
      attention_match: null,
      needs_review: false,
    },
  ];
  const semanticPage = {
    source,
    document_classification: {
      type: "model_derived_document",
      confidence: "medium",
      reason: "visible labels and table",
      evidence: ["Visible Label"],
      needs_review: false,
    },
    raw_content: rawContent,
    field_candidates: fields,
    attention_field_matches: [{
      name: "Attention Target",
      aliases: ["Attention Target"],
      status: "matched",
      matched: true,
      candidates: [fields[0]],
      needs_review: false,
    }],
    unmapped_fields: [],
    orphan_values: [],
    entities: [{
      id: "ent_00001",
      type: "model_entity",
      name: "ABC123",
      value: "ABC123",
      confidence: "high",
      evidence: { page: 1, raw_text: "Visible Label: ABC123" },
      source,
    }],
    relationships: [{
      id: "rel_00001",
      type: "field_mentions_entity",
      from: "field_p1_1",
      to: "ent_00001",
      confidence: "high",
      evidence: { page: 1, raw_text: "Visible Label: ABC123" },
      source,
    }],
    integrated_records: [],
    review_issues: [],
  };
  return {
    success: true,
    schema: "lossless_document_v1",
    source_path: source.source_path,
    pages: [{
      page: 1,
      raw_markdown: rawContent.raw_markdown,
      raw_html: "",
      text_items: rawContent.text_items,
      tables: rawContent.tables,
      field_candidates: fields.map((field) => ({
        name: field.field_name_model,
        label: field.label_original,
        value: field.value_original,
        confidence: field.confidence,
        needs_review: field.needs_review,
      })),
      mapped_fields: {},
      unmapped_fields: [],
      orphan_values: [],
      uncertain_tokens: [],
      review_required: false,
    }],
    finalJson: {},
    universal_schema: "universal_document_semantics_v2",
    extraction_policy: {
      preserve_all: true,
      extract_all_fields: true,
      attention_fields_are_hints_only: true,
      single_visual_model: true,
      model: "qwen3-vl-plus",
      generated_at: "2026-06-14 04:30:00",
    },
    raw_pages: [{ source, raw_content: rawContent }],
    semantic_pages: [semanticPage],
    detected_documents: [{
      document_id: "doc_00001",
      source_file: source.source_file,
      document_type: "model_derived_document",
      confidence: "medium",
      pages: [1],
      reason: "visible labels and table",
      evidence: ["Visible Label"],
      needs_review: false,
    }],
    entities: semanticPage.entities,
    relationships: semanticPage.relationships,
    integrated_records: [{
      record_id: "page_1",
      record_type: "model_derived_document",
      source_files: [source.source_file],
      source_pages: [1],
      confidence: "medium",
      fields: {
        attention_target: "ABC123",
        other_visible_field: "XYZ789",
      },
      needs_review: false,
    }],
    review_issues: [],
    review_required: false,
    quality_gate: { passed: true },
    stats: {
      totalApiCalls: 1,
      totalTokens: 0,
      elapsedMs: 1,
      pageCount: 1,
    },
  };
}

function worksheetRows(worksheet) {
  const header = [];
  worksheet.getRow(1).eachCell((cell, colNumber) => {
    header[colNumber] = String(cell.value || "");
  });
  const rows = [];
  worksheet.eachRow((row, rowNumber) => {
    if (rowNumber === 1) return;
    const item = {};
    row.eachCell((cell, colNumber) => {
      const key = header[colNumber];
      if (key) item[key] = cell.value;
    });
    rows.push(item);
  });
  return rows;
}
