/**
 * Vision-MCP v8.2: Second-Pass Character-Level Correction
 *
 * Two-stage extraction: raw extraction → second-pass LLM correction.
 * Uses a lightweight model (qwen3-vl-flash) to verify and correct
 * field values against known OCR confusion patterns.
 *
 * This addresses common OCR errors (STAK→ETAK, 852→552, S→E, K→X, O→0)
 * that cannot be fixed by simple pattern-based post-processing.
 */

import OpenAI from "openai";
import type { FieldSpec } from "../config/types.js";
import {
  SECOND_PASS_MODEL,
  API_KEY,
  BASE_URL,
} from "../config/constants.js";

// ---- Text-context OCR confusion pairs (alphabetic text) ----
// These are common visually-similar character confusions in OCR for text fields.
// The LLM is asked to verify each candidate against the original image.

const TEXT_CONFUSION_PAIRS: Array<[string, string, string]> = [
  // [correct, common_ocr_error, context_hint]
  ["S", "E", "uppercase S/E confusion - check stroke shape"],
  ["E", "S", "uppercase E/S confusion - check middle bar"],
  ["K", "X", "uppercase K/X confusion - check diagonal strokes"],
  ["X", "K", "uppercase X/K confusion - check intersection"],
  ["O", "0", "letter O vs digit 0 - check context (text vs number)"],
  ["0", "O", "digit 0 vs letter O - check context (number vs text)"],
  ["I", "1", "letter I vs digit 1 - check context"],
  ["1", "I", "digit 1 vs letter I - check context"],
  ["l", "1", "lowercase l vs digit 1 - check shape"],
  ["B", "8", "letter B vs digit 8 - check loops"],
  ["8", "B", "digit 8 vs letter B - check context"],
  ["Z", "2", "letter Z vs digit 2 - check angles"],
  ["2", "Z", "digit 2 vs letter Z - check context"],
  ["G", "6", "letter G vs digit 6 - check bottom curve"],
  ["6", "G", "digit 6 vs letter G - check context"],
  ["5", "S", "digit 5 vs letter S - check top bar"],
  ["S", "5", "letter S vs digit 5 - check context"],
  ["D", "O", "letter D vs letter O - check straight edge"],
  ["C", "G", "letter C vs letter G - check closure"],
];

function buildSecondPassPrompt(
  fieldValues: Record<string, string>,
  fieldSpecs: FieldSpec[]
): string {
  const entries = Object.entries(fieldValues)
    .filter(([, v]) => v && v.length > 0)
    .map(([name, value]) => {
      const spec = fieldSpecs.find((f) => f.name === name);
      const fmt = spec?.formatHint ? ` (expected format: ${spec.formatHint})` : "";
      const ex = spec?.example ? ` (example: ${spec.example})` : "";
      const allowed = spec?.allowedValues?.length
        ? ` (allowed values: ${spec.allowedValues.join(", ")})`
        : "";
      return `  "${name}": "${value}"${fmt}${ex}${allowed}`;
    });

  const confusionHints = TEXT_CONFUSION_PAIRS
    .map(([correct, error, hint]) => `  - "${correct}" is often misread as "${error}" (${hint})`)
    .join("\n");

  return `You are a precision OCR correction specialist. Review the following extracted field values against the original document image.

Common OCR confusion patterns to check:
${confusionHints}

Current extracted values:
${entries.join("\n")}

Rules:
1. Compare each value against what you actually see in the image
2. If a character looks ambiguous, check against the confusion pairs above
3. For company names, verify against the visible text on the document header/signature area
4. For phone/fax numbers, verify all digits carefully - especially area codes
5. For addresses, verify building numbers and street names
6. Do NOT change values that match the document - only correct real OCR errors
7. If unsure, keep the original value
8. Output ONLY valid JSON with corrected values: {"field_name": "corrected_value", ...}
9. If no correction needed for a field, still include it with the original value
10. For fields that are completely correct, do not modify them at all

IMPORTANT: The original document text is the source of truth. Do not fabricate or guess.`;
}

export async function secondPassCorrect(
  fieldValues: Record<string, string>,
  fieldSpecs: FieldSpec[],
  imageBase64: string,
  imageMime?: string
): Promise<{ corrected: Record<string, string>; corrections: string[]; apiCalls: number }> {
  const tStart = Date.now();
  const corrections: string[] = [];
  const mime = imageMime || "image/png";

  // Skip if no values to check
  const nonEmptyCount = Object.values(fieldValues).filter((v) => v && v.length > 0).length;
  if (nonEmptyCount === 0) {
    console.error("[second-pass] No non-empty values to verify - skipping");
    return { corrected: { ...fieldValues }, corrections, apiCalls: 0 };
  }

  console.error(`[second-pass] Verifying ${nonEmptyCount} field(s) with ${SECOND_PASS_MODEL}...`);

  try {
    const client = new OpenAI({
      apiKey: API_KEY,
      baseURL: BASE_URL,
      timeout: 120_000,
      maxRetries: 1,
    });

    const prompt = buildSecondPassPrompt(fieldValues, fieldSpecs);

    const r = await client.chat.completions.create({
      model: SECOND_PASS_MODEL,
      max_tokens: 4096,
      temperature: 0,
      // @ts-ignore
      enable_thinking: true, // v8.3: thinking ON for second-pass correction
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: `data:${mime};base64,${imageBase64}`,
              },
            },
            { type: "text", text: prompt },
          ],
        },
      ],
      response_format: { type: "json_object" },
    });

    const rawText = r.choices[0]?.message?.content || "{}";
    let corrected: Record<string, string>;

    try {
      corrected = JSON.parse(rawText);
    } catch {
      console.error("[second-pass] Failed to parse JSON response, keeping original values");
      return { corrected: { ...fieldValues }, corrections: ["second-pass JSON parse failed"], apiCalls: 1 };
    }

    // Merge: only use corrected values that differ from originals
    const final: Record<string, string> = {};
    for (const [name, originalValue] of Object.entries(fieldValues)) {
      const correctedValue = corrected[name];
      if (correctedValue && correctedValue !== originalValue && correctedValue.trim().length > 0) {
        const origT = originalValue.trim();
        const corrT = correctedValue.trim();
        if (corrT !== origT) {
          const firstCharChanged = origT.charAt(0) !== corrT.charAt(0);
          const lenDiff = Math.abs(origT.length - corrT.length);
          if (firstCharChanged && (lenDiff > 0 || origT.length > 4)) {
            console.error("[second-pass] Skipping " + name + ": first-char change suggests overcorrection");
            final[name] = originalValue;
            corrections.push(name + ": " + originalValue + " -> " + correctedValue + " SKIPPED (overcorrection risk)");
          } else {
            final[name] = correctedValue;
            corrections.push(name + ": " + originalValue + " -> " + correctedValue + "");
            console.error("[second-pass] Corrected " + name + ": " + originalValue + " -> " + correctedValue + "");
          }
        } else {
          final[name] = originalValue;
        }
      } else {
        final[name] = originalValue;
      }
    }

    // Also include any new fields from the correction pass
    for (const [name, value] of Object.entries(corrected)) {
      if (!(name in final) && value) {
        final[name] = value;
      }
    }

    const elapsed = Date.now() - tStart;
    console.error(
      `[second-pass] Complete: ${corrections.length} corrections in ${elapsed}ms, input_tokens=${r.usage?.prompt_tokens || "?"}, output_tokens=${r.usage?.completion_tokens || "?"}`
    );

    return { corrected: final, corrections, apiCalls: 1 };
  } catch (err: any) {
    console.error(`[second-pass] Failed: ${err.message}. Keeping original values.`);
    return { corrected: { ...fieldValues }, corrections: [`second-pass failed: ${err.message}`], apiCalls: 0 };
  }
}
