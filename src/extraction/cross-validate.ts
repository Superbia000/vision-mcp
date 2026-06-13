/**
 * Vision-MCP v8: L3 - Enhanced Cross-Validation Engine
 *
 * v8 improvements:
 * - Consensus entropy scoring per field
 * - Smart fallback with label/value swap detection
 * - JSON structured output validation
 * - More granular confidence scoring
 */

import type { VisionProvider } from "../providers/base.js";
import type { FieldSpec, LocatedField } from "../config/types.js";

const VALIDATION_PROMPT = `You are a document data validator. Compare the full-page extraction with per-field extractions.
For each field:
1. If both extractions agree -> keep the value, mark verified=true
2. If they differ -> prefer the per-field extraction (higher resolution), mark verified=true
3. If one extraction is empty -> use the non-empty one, mark verified=false
4. Apply format validation (dates, numbers, etc.)
5. For each field discrepancy, note what was corrected

Output final validated JSON with this structure:
{
  "fields": {
    "field_name": {
      "value": "final value",
      "confidence": "high|medium|low",
      "verified": true|false,
      "discrepancy_note": "if any"
    }
  }
}`;

/**
 * Check if we can skip Layer 3 validation (all fields agree, high confidence).
 */
export function shouldSkipValidation(
  layoutFields: LocatedField[],
  perFieldResults: Map<string, { value: string; confidence: string }>,
  fieldSpecs: FieldSpec[]
): boolean {
  for (const spec of fieldSpecs) {
    const layoutField = layoutFields.find(
      (f) =>
        f.name.toLowerCase() === spec.name.toLowerCase() ||
        f.label.toLowerCase().includes(spec.labelPattern.toLowerCase())
    );
    const pf = perFieldResults.get(spec.name);
    if (layoutField && pf && layoutField.value !== pf.value) {
      return false;
    }
    if (!pf || pf.confidence === "low") {
      return false;
    }
  }
  return true;
}

/**
 * Smart fallback: detect label/value swaps and prefer correct value.
 */
function smartFallback(
  layoutFields: LocatedField[],
  perFieldResults: Map<string, { value: string; confidence: string }>
): Record<string, { value: string; confidence: string; verified: boolean }> {
  const result: Record<string, { value: string; confidence: string; verified: boolean }> = {};
  for (const lf of layoutFields) {
    const pf = perFieldResults.get(lf.name);
    if (pf && lf.label && pf.value.trim().toLowerCase() === lf.label.trim().toLowerCase()) {
      result[lf.name] = {
        value: lf.value,
        confidence: lf.confidence,
        verified: true,
      };
      console.error(
        `[L3] Label/value swap for "${lf.name}": per-field="${pf.value}" matches label, using layout="${lf.value}"`
      );
    } else if (pf) {
      result[lf.name] = { value: pf.value, confidence: pf.confidence, verified: true };
    } else {
      result[lf.name] = { value: lf.value, confidence: lf.confidence || "low", verified: false };
    }
  }
  return result;
}

/** v8: Compute simple consensus entropy from field results */
export function computeFieldConsensusEntropy(
  layoutFields: LocatedField[],
  perFieldResults: Map<string, { value: string; confidence: string }>,
  fieldSpecs: FieldSpec[]
): Record<string, number> {
  const entropy: Record<string, number> = {};

  for (const spec of fieldSpecs) {
    const layoutField = layoutFields.find(
      (f) =>
        f.name.toLowerCase() === spec.name.toLowerCase() ||
        f.label.toLowerCase().includes(spec.labelPattern.toLowerCase())
    );
    const pf = perFieldResults.get(spec.name);

    if (!layoutField && !pf) {
      entropy[spec.name] = 1.0; // Maximum entropy: no data
      continue;
    }

    const l1Value = layoutField?.value || "";
    const l2Value = pf?.value || "";
    const l1Conf = layoutField?.confidence || "low";
    const l2Conf = pf?.confidence || "low";

    // Simple edit-distance based entropy between L1 and L2 results
    const dist = editDistance(l1Value, l2Value);
    const maxLen = Math.max(l1Value.length, l2Value.length, 1);
    const normalizedDist = dist / maxLen;

    // Confidence penalty
    let confPenalty = 0;
    if (l1Conf === "low") confPenalty += 0.3;
    else if (l1Conf === "medium") confPenalty += 0.15;
    if (l2Conf === "low") confPenalty += 0.3;
    else if (l2Conf === "medium") confPenalty += 0.15;

    entropy[spec.name] = Math.min(1.0, normalizedDist + confPenalty);
  }

  return entropy;
}

/** Simple Levenshtein distance for entropy calculation */
function editDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }

  return dp[m][n];
}

export async function crossValidate(
  provider: VisionProvider,
  model: string,
  layoutFields: LocatedField[],
  perFieldResults: Map<string, { value: string; confidence: string }>,
  fieldSpecs?: FieldSpec[]
): Promise<Record<string, { value: string; confidence: string; verified: boolean }>> {
  const tStart = Date.now();

  const fullPageExtraction = layoutFields
    .map((f) => `  "${f.name}": "${f.value}" (confidence: ${f.confidence})`)
    .join("\n");

  const perFieldExtraction = Array.from(perFieldResults.entries())
    .map(([name, r]) => `  "${name}": "${r.value}" (confidence: ${r.confidence})`)
    .join("\n");

  const formatRules =
    fieldSpecs
      ?.map(
        (f) =>
          `  "${f.name}": ${f.formatHint ? `format: "${f.formatHint}"` : "no format constraint"}`
      )
      .join("\n") || "";

  try {
    const r = await provider.chat({
      model,
      max_tokens: 4096,
      temperature: 0,
      enable_thinking: true, // v8.3: thinking ON for validation
      messages: [
        {
          role: "user",
          content: `${VALIDATION_PROMPT}\n\n## Full-page extraction:\n${fullPageExtraction}\n\n## Per-field extraction (higher resolution):\n${perFieldExtraction}\n\n## Format rules:\n${formatRules}\n\nCross-validate and output final JSON. The JSON keyword must appear.`,
        },
      ],
      response_format: { type: "json_object" },
    });

    const content = r.text;
    const parsed = JSON.parse(content);
    console.error(
      `[L3] Cross-validation: ${Object.keys(parsed.fields || {}).length} fields in ${Date.now() - tStart}ms`
    );

    // Post-process to fix label/value swaps
    const validated = parsed.fields || {};
    for (const [name, vf] of Object.entries(validated) as [string, any][]) {
      const layoutField = layoutFields.find((f) => f.name === name);
      if (
        layoutField &&
        vf.value &&
        layoutField.label &&
        vf.value.trim().toLowerCase() === layoutField.label.trim().toLowerCase()
      ) {
        vf.value = layoutField.value;
        console.error(`[L3] Post-corrected label/value swap for "${name}"`);
      }
    }
    return validated;
  } catch (err: any) {
    console.error(`[L3] Cross-validation failed: ${err.message}. Using smart fallback.`);
    return smartFallback(layoutFields, perFieldResults);
  }
}
