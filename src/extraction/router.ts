/**
 * Vision-MCP v9: Extraction Strategy Router
 *
 * Routes extraction to the optimal strategy based on document type detection:
 * - scan / photo / handwriting / mixed → full-page OCR with adaptive prompts + dual-pass annealing
 * - table → L1-L5 layered extraction (needs bbox for column alignment)
 *
 * v9 changes:
 * - Adaptive reading prompts (teach HOW to read, not WHERE)
 * - Dual-pass temperature annealing (0 → 0.03, early stop when all high confidence)
 * - Removed second-pass correction (now disabled by default)
 */

import type { VisionProvider } from "../providers/base.js";
import type { FieldSpec, LayeredExtractionConfig, DocumentType } from "../config/types.js";
import {
  VL_HIGH_RES_ENABLED, ENABLE_STRUCTURED_OUTPUT,
  ENABLE_POST_PROCESS, ENABLE_CROSS_FIELD,
  MULTIPASS_TEMPERATURES,
} from "../config/constants.js";
import { preprocessScanned } from "../preprocessing/pipeline.js";
import { buildFieldExtractionPrompt } from "./prompts.js";
import { applyFormatCorrection, validateCrossField } from "./post-process.js";

/** Strategy names */
export type ExtractionStrategy = "auto" | "full-page" | "layered";

/** Result of strategy selection */
export interface StrategyDecision {
  strategy: ExtractionStrategy;
  docType: DocumentType;
  reason: string;
}

/**
 * Decide which extraction strategy to use based on document type and config.
 */
export function decideStrategy(
  docType: DocumentType,
  forcedStrategy?: string
): StrategyDecision {
  const envStrategy = process.env.VISION_EXTRACTION_STRATEGY as ExtractionStrategy | undefined;
  const effectiveForced = forcedStrategy || envStrategy;

  if (effectiveForced === "full-page") {
    return { strategy: "full-page", docType, reason: "forced by config" };
  }
  if (effectiveForced === "layered") {
    return { strategy: "layered", docType, reason: "forced by config" };
  }

  // Auto-routing: all non-table docs use full-page for better single-model accuracy
  const scannedStrategy = process.env.VISION_SCANNED_STRATEGY || "full-page";

  if (docType === "scan" || docType === "photo" || docType === "handwriting" || docType === "mixed") {
    return {
      strategy: scannedStrategy as ExtractionStrategy,
      docType,
      reason: `docType=${docType} -> ${scannedStrategy}`,
    };
  }

  // table -> use layered
  return { strategy: "layered", docType, reason: `docType=${docType} -> layered` };
}

/** v9: Match model output field to user spec by label pattern (pipe-separated) */
function matchField(rawField: any, spec: FieldSpec): boolean {
  const rawName = String(rawField.name || rawField.key || rawField.field || "").toLowerCase();
  if (rawName && rawName === spec.name.toLowerCase()) return true;

  const label = (rawField.label || "").toLowerCase();
  const patternStr = spec.labelPattern.toLowerCase();
  const patterns = patternStr.split("|").map(p => p.trim()).filter(p => p.length > 0);
  for (const pattern of patterns) {
    if (label.includes(pattern) || pattern.includes(label)) return true;
  }
  return (rawField.name || "").toLowerCase() === spec.name.toLowerCase();
}

function findRawField(fields: any[], spec: FieldSpec): any | undefined {
  return fields.find((f: any) => matchField(f, spec));
}

function rawFieldValue(rawField: any): string {
  if (!rawField) return "";
  const value = rawField.value ?? rawField.text ?? rawField.raw ?? "";
  return typeof value === "string" ? value.trim() : String(value ?? "").trim();
}

function compactFieldValue(value: string): string {
  return value.toUpperCase().normalize("NFKC").replace(/[\s,.:;'"`_\/\\()\[\]#]+/g, "");
}

export function fieldValueLooksValid(value: string, spec: FieldSpec): boolean {
  const trimmed = value.trim();
  if (!trimmed) return spec.required === false;
  if (trimmed.includes("[?]")) return false;
  void spec;
  return true;
}

/**
 * v9: Full-page OCR extraction with adaptive reading prompts.
 * Uses the single primary model (qwen3-vl-plus), no dedicated OCR model.
 */
export async function fullPageExtract(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  config: LayeredExtractionConfig,
  crossPageHint?: string
): Promise<Record<string, any>> {
  const tStart = Date.now();
  const docType = (config as any).docType || "scan";
  console.error(`[full-page v9] Starting extraction for ${fieldSpecs.length} fields (docType=${docType})...`);

  // Preprocess with scanned-optimized pipeline
  const pp = await preprocessScanned(imageBuffer);
  const ppBuf = Buffer.from(pp.buffer);
  const ppBase64 = ppBuf.toString("base64");

  // v9: Build adaptive reading prompt (teaches HOW to read, not WHERE)
  const prompt = buildFieldExtractionPrompt(docType, fieldSpecs, crossPageHint);

  const r = await provider.chat({
    model: config.primaryModel,
    max_tokens: 16384,
    temperature: 0,
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${pp.mime};base64,${ppBase64}` } },
        { type: "text", text: prompt },
      ],
    }],
    vl_high_resolution_images: VL_HIGH_RES_ENABLED,
    ...(ENABLE_STRUCTURED_OUTPUT ? { response_format: { type: "json_object" } } : {}),
  });

  let result: Record<string, any> = {};
  try {
    result = JSON.parse(r.text || "{}");
  } catch {
    console.error("[full-page v9] Failed to parse JSON response, using raw text");
    result = { _raw: r.text || "" };
  }

  // Normalize to standard format: match fieldSpecs by label pattern
  const finalJson: Record<string, any> = {};
  const rawFields: any[] = result.fields || [];

  for (const spec of fieldSpecs) {
    const rawField = findRawField(rawFields, spec);
    const value = rawFieldValue(rawField);
    const isValid = fieldValueLooksValid(value, spec);

    if (rawField && value) {
      finalJson[spec.name] = {
        value,
        confidence: isValid ? (rawField.confidence || "medium") : "low",
        verified: rawField.confidence === "high" && isValid,
        source: "full-page OCR",
        ...(!isValid ? { validation_warning: "value does not match requested field shape" } : {}),
      };
    } else if (rawField && typeof rawField === "string") {
      finalJson[spec.name] = {
        value: rawField,
        confidence: "medium",
        verified: false,
        source: "full-page OCR",
      };
    } else {
      finalJson[spec.name] = {
        value: "",
        confidence: "low",
        verified: false,
        error: "field not found in response",
      };
    }
  }

  // L5 Post-processing: safe normalizations only (no character mapping)
  if (ENABLE_POST_PROCESS) {
    for (const spec of fieldSpecs) {
      const entry = finalJson[spec.name];
      if (!entry || !entry.value) continue;
      const trimmed = entry.value.trim();
      const normalized = trimmed.replace(/\//g, "-");
      if (normalized !== entry.value) {
        finalJson[spec.name] = { ...entry, value: normalized, post_processed: true };
      }
    }
  }

  const elapsed = Date.now() - tStart;
  console.error(
    `[full-page v9] Complete: ${fieldSpecs.length} fields in ${elapsed}ms, steps=[${pp.appliedSteps.slice(0, 3).join(" → ")}...]`
  );

  Object.defineProperty(finalJson, "_visionUsage", {
    value: [{ stage: "full_page_extract", model: config.primaryModel, input_tokens: r.it, output_tokens: r.ot }],
    enumerable: false,
  });

  return finalJson;
}

/**
 * v9: Dual-pass temperature annealing extraction.
 *
 * Pass 1: T=0, full extraction.
 *   → If all fields have confidence=high and no [?] marks → return immediately.
 * Pass 2: T=0.03, only for uncertain fields.
 *   → Merge: unanimous → adopt; edit distance ≤1 → take longer version;
 *     large diff → keep both, mark low.
 */
export async function multiPassExtract(
  provider: VisionProvider,
  imageBuffer: Buffer,
  imageMime: string,
  fieldSpecs: FieldSpec[],
  config: LayeredExtractionConfig,
  passes: number = 2
): Promise<{
  finalJson: Record<string, any>;
  passes: Record<string, any>[];
  consensus: Record<string, { values: string[]; agreement: number; unanimous: boolean }>;
  usage?: { stage: string; model: string; input_tokens?: number; output_tokens?: number }[];
}> {
  const temperatures = MULTIPASS_TEMPERATURES
    .split(",")
    .map(Number)
    .slice(0, passes);

  while (temperatures.length < passes) temperatures.push(0);

  console.error(`[multi-pass v9] Dual-pass annealing: temps=[${temperatures.join(", ")}]`);

  const allResults: Record<string, any>[] = [];
  const usage: { stage: string; model: string; input_tokens?: number; output_tokens?: number }[] = [];
  const tStart = Date.now();
  const docType = (config as any).docType || "scan";

  // ── Pass 1: T=0 ──
  console.error(`[multi-pass v9] Pass 1/2 (T=${temperatures[0]})...`);
  const pp = await preprocessScanned(imageBuffer);
  const ppBuf = Buffer.from(pp.buffer);
  const ppBase64 = ppBuf.toString("base64");

  const prompt1 = buildFieldExtractionPrompt(docType, fieldSpecs);

  const r1 = await provider.chat({
    model: config.primaryModel,
    max_tokens: 16384,
    temperature: temperatures[0],
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${pp.mime};base64,${ppBase64}` } },
        { type: "text", text: prompt1 },
      ],
    }],
    vl_high_resolution_images: VL_HIGH_RES_ENABLED,
    ...(ENABLE_STRUCTURED_OUTPUT ? { response_format: { type: "json_object" } } : {}),
  });

  try {
    allResults.push(JSON.parse(r1.text || "{}"));
  } catch {
    allResults.push({ _raw: r1.text || "" });
  }
  usage.push({ stage: "multi_pass_1", model: config.primaryModel, input_tokens: r1.it, output_tokens: r1.ot });

  // ── Check if Pass 2 is needed ──
  const pass1Fields: any[] = allResults[0]?.fields || [];
  let needsPass2 = false;
  const uncertainFields: Set<string> = new Set();

  for (const spec of fieldSpecs) {
    const rawField = findRawField(pass1Fields, spec);
    const value = rawFieldValue(rawField);
    const conf = rawField?.confidence || "low";
    const hasUncertainty = value.includes("[?]") || !value || conf === "low" || !fieldValueLooksValid(value, spec);

    if (hasUncertainty) {
      needsPass2 = true;
      uncertainFields.add(spec.name);
    }
  }

  if (!needsPass2) {
    console.error("[multi-pass v9] All fields high confidence - skipping Pass 2 (early stop)");
    const finalJson: Record<string, any> = {};
    const consensus: Record<string, { values: string[]; agreement: number; unanimous: boolean }> = {};

    for (const spec of fieldSpecs) {
      const rawField = findRawField(pass1Fields, spec);
      const val = rawFieldValue(rawField);
      const isValid = fieldValueLooksValid(val, spec);
      finalJson[spec.name] = {
        value: val,
        confidence: isValid ? (rawField?.confidence || "medium") : "low",
        verified: isValid,
        source: "multi-pass (1 pass, early stop)",
        ...(!isValid ? { validation_warning: "value does not match requested field shape" } : {}),
      };
      consensus[spec.name] = { values: [val], agreement: 1.0, unanimous: true };
    }

    const elapsed = Date.now() - tStart;
    console.error(`[multi-pass v9] Complete: 1 pass in ${elapsed}ms (early stop)`);
    return { finalJson, passes: allResults, consensus, usage };
  }

  // ── Pass 2: T=0.03, only for uncertain fields ──
  console.error(
    `[multi-pass v9] Pass 2/2 (T=${temperatures[1] || 0.03}) for ${uncertainFields.size} uncertain field(s): [${[...uncertainFields].join(", ")}]`
  );

  const uncertainNames = [...uncertainFields];
  const focusedLabels = uncertainNames.map((name) => {
    const spec = fieldSpecs.find((s) => s.name === name);
    return spec?.labelPattern || name;
  });

  const focusedSpecs = uncertainNames
    .map((name) => fieldSpecs.find((s) => s.name === name))
    .filter(Boolean) as FieldSpec[];
  const prompt2 = buildFieldExtractionPrompt(docType, focusedSpecs) +
    `\n\n重要：請特別仔細地重新閱讀以下欄位，逐字確認每個字元：${focusedLabels.join("、")}。`;

  const r2 = await provider.chat({
    model: config.primaryModel,
    max_tokens: 16384,
    temperature: temperatures[1] || 0.03,
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${pp.mime};base64,${ppBase64}` } },
        { type: "text", text: prompt2 },
      ],
    }],
    vl_high_resolution_images: VL_HIGH_RES_ENABLED,
    ...(ENABLE_STRUCTURED_OUTPUT ? { response_format: { type: "json_object" } } : {}),
  });

  try {
    allResults.push(JSON.parse(r2.text || "{}"));
  } catch {
    allResults.push({ _raw: r2.text || "" });
  }
  usage.push({ stage: "multi_pass_2_uncertain", model: config.primaryModel, input_tokens: r2.it, output_tokens: r2.ot });

  // ── Merge results ──
  const finalJson: Record<string, any> = {};
  const consensus: Record<string, { values: string[]; agreement: number; unanimous: boolean }> = {};

  for (const spec of fieldSpecs) {
    const values: string[] = [];
    for (const result of allResults) {
      const fields: any[] = result?.fields || [];
      const rawField = findRawField(fields, spec);
      const value = rawFieldValue(rawField);
      if (value) values.push(value);
    }

    const validValues = values.filter((v) => fieldValueLooksValid(v, spec));
    const candidateValues = validValues.length ? validValues : values;
    const unique = [...new Set(candidateValues)];
    const counts = new Map<string, number>();
    for (const v of candidateValues) counts.set(v, (counts.get(v) || 0) + 1);
    const topCount = Math.max(0, ...counts.values());
    const agreement = candidateValues.length > 0 ? topCount / candidateValues.length : 0;
    const unanimous = unique.length === 1 && candidateValues.length >= allResults.length && fieldValueLooksValid(unique[0], spec);

    let bestValue = "";
    if (unanimous) {
      bestValue = unique[0];
    } else if (candidateValues.length >= 2) {
      const d = editDistance(candidateValues[0], candidateValues[1]);
      if (d <= 1) {
        bestValue = candidateValues[0].length >= candidateValues[1].length ? candidateValues[0] : candidateValues[1];
      } else {
        bestValue = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || candidateValues[0];
      }
    } else {
      bestValue = candidateValues[0] || "";
    }

    consensus[spec.name] = { values: unique, agreement, unanimous };
    finalJson[spec.name] = {
      value: bestValue,
      confidence: unanimous ? "high" : agreement >= 0.5 ? "medium" : "low",
      verified: unanimous,
      source: `multi-pass (${allResults.length} passes, agreement=${(agreement * 100).toFixed(0)}%)`,
      ...(!fieldValueLooksValid(bestValue, spec) ? { validation_warning: "value does not match requested field shape" } : {}),
      ...(agreement < 1 ? { discrepancy_note: `${unique.length} distinct values: [${unique.join(" | ")}]` } : {}),
    };
  }

  const elapsed = Date.now() - tStart;
  console.error(
    `[multi-pass v9] Complete: ${allResults.length} passes in ${elapsed}ms, ` +
    `${Object.values(consensus).filter((c) => c.unanimous).length}/${fieldSpecs.length} unanimous`
  );

  return { finalJson, passes: allResults, consensus, usage };
}

/** Levenshtein distance for merge comparison */
function editDistance(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}
