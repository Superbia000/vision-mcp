/**
 * Vision-MCP v8: Post-processing and format-aware OCR correction
 *
 * Handles:
 * - Format-based character correction (date, number, currency patterns)
 * - Common OCR confusion pairs (0/O, 1/l, S/5, etc.)
 * - Cross-field validation (totals, consistency checks)
 * - Chinese character confusion dictionary
 */

import type { FieldSpec } from "../config/types.js";

// ---- Common OCR confusion character pairs ----

const OCR_CORRECTIONS: Record<string, Record<string, string>> = {
  // Digit context corrections (when format hint says "number" or "date")
  number: { O: "0", o: "0", l: "1", I: "1", S: "5", s: "5", B: "8", Z: "2", z: "2", T: "7" },
  // Date-specific
  date: { O: "0", o: "0", l: "1", I: "1", S: "5", s: "5" },
  // Currency-specific
  currency: { O: "0", o: "0", l: "1", I: "1", S: "5", s: "5", B: "8" },
  // ID card numbers
  id_card: { O: "0", o: "0", l: "1", I: "1", S: "5", s: "5", B: "8", Z: "2", z: "2" },
  // Phone numbers
  phone: { O: "0", o: "0", l: "1", I: "1" },
  // Email addresses
  email: { O: "0", o: "0" }, // Email mostly preserves alpha, only fix numbers
  // v8.2: Text context corrections (alphabetic fields like names, addresses)
  text: {
    // Common uppercase letter confusions
    S: "E?", E: "S?", K: "X?", X: "K?",
    D: "O?", O: "D?", C: "G?", G: "C?",
    B: "8?", 8: "B?",
    // Common letter/digit confusions in text context
    0: "O?", 1: "I?", 2: "Z?", 5: "S?",
    I: "1?", l: "1?",
  },
};

/** Chinese character confusion pairs (visually similar) */
const CHINESE_CONFUSION: Array<[string, string]> = [
  ["已", "己"], ["未", "末"], ["日", "曰"], ["土", "士"],
  ["千", "干"], ["人", "入"], ["八", "入"], ["大", "太"],
  ["天", "夫"], ["王", "玉"], ["甲", "由"], ["田", "由"],
  ["牛", "午"], ["午", "牛"], ["貝", "見"], ["見", "貝"],
  ["准", "淮"], ["淮", "准"], ["侯", "候"], ["候", "侯"],
];

// ---- Format-aware correction ----

/** Apply OCR corrections based on format hint */
export function applyFormatCorrection(
  value: string,
  formatHint?: string,
  formatType?: "date" | "number" | "email" | "phone" | "currency" | "id_card"
): { corrected: string; corrections: string[] } {
  const corrections: string[] = [];
  let corrected = value;

  if (!value) return { corrected, corrections };

  // Determine correction set from format
  const hint = (formatHint || "").toLowerCase();
  let correctionSet: Record<string, string> | undefined;

  if (formatType === "date" || hint.includes("date") || hint.includes("yyyy") || hint.includes("mm-dd")) {
    correctionSet = OCR_CORRECTIONS.date;
    // Also normalize date separators
    const orig = corrected;
    corrected = corrected.replace(/\//g, "-");
    if (corrected !== orig) corrections.push("normalized date separators");
  } else if (formatType === "currency" || hint.includes("price") || hint.includes("amount") || hint.includes("currency") || hint.includes("¥") || hint.includes("$")) {
    correctionSet = OCR_CORRECTIONS.currency;
  } else if (formatType === "id_card" || hint.includes("id") || hint.includes("identity")) {
    correctionSet = OCR_CORRECTIONS.id_card;
  } else if (formatType === "phone" || hint.includes("phone") || hint.includes("tel")) {
    correctionSet = OCR_CORRECTIONS.phone;
  } else if (formatType === "email" || hint.includes("email") || hint.includes("@")) {
    correctionSet = OCR_CORRECTIONS.email;
  } else if (formatType === "number" || hint.includes("number") || hint.includes("digit")) {
    correctionSet = OCR_CORRECTIONS.number;
  } else if (formatType === "text" || hint.includes("text") || hint.includes("name") || hint.includes("address") || hint.includes("company")) {
    correctionSet = OCR_CORRECTIONS.text;
  }

  // Apply character-level corrections (v8.2: flag uncertain corrections with ?)
  if (correctionSet) {
    let charCorrected = "";
    for (const ch of corrected) {
      const replacement = correctionSet[ch];
      if (replacement) {
        // If replacement ends with ?, it is uncertain - keep original but note
        if (replacement.endsWith("?")) {
          charCorrected += ch;
          corrections.push(`'${ch}' might be '${replacement.replace("?","")}' (uncertain)`);
        } else {
          charCorrected += replacement;
          corrections.push(`'${ch}' -> '${replacement}'`);
        }
      } else {
        charCorrected += ch;
      }
    }
    corrected = charCorrected;
  }

  // Trim whitespace and normalize
  const trimmed = corrected.trim();
  if (trimmed !== corrected) {
    corrected = trimmed;
    corrections.push("trimmed whitespace");
  }

  // Remove leading/trailing punctuation artifacts
  const cleaned = corrected.replace(/^[.,;:]+|[.,;:]+$/g, "");
  if (cleaned !== corrected) {
    corrected = cleaned;
    corrections.push("removed punctuation artifacts");
  }

  return { corrected, corrections };
}

// ---- Cross-field validation ----

/** Cross-field consistency checks */
export function validateCrossField(
  fields: Record<string, string>,
  fieldSpecs: FieldSpec[]
): string[] {
  const notes: string[] = [];

  // Check for context rules
  for (const spec of fieldSpecs) {
    if (spec.contextRule && fields[spec.name]) {
      const rule = spec.contextRule;
      // Parse simple rules like "total = subtotal + tax"
      const sumMatch = rule.match(/(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)/);
      if (sumMatch) {
        const [, target, a, b] = sumMatch;
        const valA = parseFloat(fields[a] || "0");
        const valB = parseFloat(fields[b] || "0");
        const valTarget = parseFloat(fields[target] || "0");
        if (!isNaN(valA) && !isNaN(valB) && !isNaN(valTarget)) {
          const expected = valA + valB;
          if (Math.abs(expected - valTarget) > 0.02) {
            notes.push(`Cross-field: ${target}=${valTarget} but ${a}+${b}=${expected} (diff=${(expected - valTarget).toFixed(2)})`);
          }
        }
      }

      // Check inequality rules like "departure != destination"
      const neMatch = rule.match(/(\w+)\s*!=\s*(\w+)/);
      if (neMatch) {
        const [, a, b] = neMatch;
        if (fields[a] && fields[b] && fields[a].trim() === fields[b].trim()) {
          notes.push(`Cross-field: ${a} equals ${b} ("${fields[a]}") but should differ`);
        }
      }
    }
  }

  return notes;
}

/** Chinese character confusion check */
export function checkChineseConfusion(value: string, formatHint?: string): string[] {
  const warnings: string[] = [];
  // Only apply to Chinese text contexts
  const hasChinese = /[\u4e00-\u9fff]/.test(value);
  if (!hasChinese) return warnings;

  for (const [a, b] of CHINESE_CONFUSION) {
    if (value.includes(a)) {
      warnings.push(`Chinese confusion: "${a}" present, could be "${b}"`);
    }
  }

  return warnings;
}


// ---- v8.3: Phone/Fax Number Format Validation ----

const PHONE_PATTERNS: Record<string, RegExp> = {
  hk: /^\(852\)\s?\d{4}\s?\d{4}$/,
  hk_alt: /^(\+?852[\s-]?)?\d{4}[\s-]?\d{4}$/,
  generic: /^\(\+?\d{1,4}\)\s?[\d\s\-]{7,15}$/,
  any: /^[+\d][\d\s\(\)\-\.]{6,20}$/,
};

/** Validate phone number against known patterns. Detects and corrects
 *  common hallucinations: +45→852 (Denmark→HK), 452→852 (transposition). */
export function validatePhoneNumber(
  value: string,
  formatHint?: string
): { valid: boolean; suggested: string; issues: string[] } {
  const issues: string[] = [];
  let cleaned = value.replace(/[\s\-]/g, "");

  if (cleaned.match(/^\(\+?45\)/)) {
    issues.push("hallucinated +45 (likely HK 852)");
    cleaned = cleaned.replace(/^\(\+?45\)/, "(852)");
  }
  if (cleaned.match(/^\(452\)/)) {
    issues.push("invalid prefix 452 (likely HK 852)");
    cleaned = cleaned.replace(/^\(452\)/, "(852)");
  }
  if (cleaned.match(/^\+?452/)) {
    issues.push("invalid prefix +452 (likely HK 852)");
    cleaned = cleaned.replace(/^\+?452/, "+852");
  }

  const hint = (formatHint || "").toLowerCase();
  const isHK = hint.includes("hk") || hint.includes("hong kong") || hint.includes("852");

  if (isHK) {
    for (const key of ["hk", "hk_alt"]) {
      if (PHONE_PATTERNS[key].test(value) || PHONE_PATTERNS[key].test(cleaned)) {
        return { valid: true, suggested: "", issues };
      }
    }
    return { valid: false, suggested: cleaned !== value ? cleaned : "", issues };
  }

  if (PHONE_PATTERNS.generic.test(value)) {
    return { valid: true, suggested: "", issues };
  }

  return { valid: false, suggested: cleaned !== value ? cleaned : "", issues };
}

/** Quick check: does this phone number look hallucinated and need re-extraction? */
export function isPhoneHallucination(value: string): boolean {
  const cleaned = value.replace(/[\s\-]/g, "");
  if (cleaned.match(/^\(\+?45\)/) || cleaned.match(/^\+45/)) return true;
  if (cleaned.match(/^\(452\)/)) return true;
  return false;
}
