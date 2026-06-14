/**
 * Vision-MCP v8: Shared tool helpers
 *
 * v8: Removed layeredSinglePage - all field extraction now routes through
 * layeredExtract() in extraction/layered.ts for unified L1-L5 pipeline.
 */

import OpenAI from "openai";
import type { FieldSpec, LayeredExtractionConfig, LayeredExtractionResult } from "../config/types.js";
import {
  MODEL, OCR_MODEL, BASE_URL, OCR_BASE_URL, API_KEY,
  ENABLE_THINKING, ENABLE_LAYERED_EXTRACTION, ENABLE_SELF_CONSISTENCY,
  CE_VOTES,
} from "../config/constants.js";
import { layeredExtract } from "../extraction/layered.js";
import { weightedConsistencyVote } from "../extraction/ensemble.js";
import { IS_QWEN } from "../config/constants.js";

/**
 * Prompt enhancement: Qwen-specific, language hints
 */
export function enhancePrompt(prompt: string): string {
  let enhanced = prompt;

  if (IS_QWEN && !/qwenvl/i.test(prompt)) {
    const docKeywords = /OCR|extract|document|table|parse|markdown|html/i;
    if (docKeywords.test(prompt)) {
      console.error(`[prompt] Qwen document extraction - using optimized prompt`);
    }
  }

  const hasChinese = /[\u4e00-\u9fff]/.test(prompt);
  const shouldAppendLanguageHint = hasChinese && !/(Traditional Chinese|Chinese|language)/i.test(prompt);
  if (shouldAppendLanguageHint) {
    enhanced = enhanced +
      "\nPlease answer in Traditional Chinese when field labels or source text are Chinese.";
    console.error(`[prompt] Auto-appended Traditional Chinese language hint`);
  }

  return enhanced;
}

/**
 * v8: Unified single-page extraction - always routes through layeredExtract.
 */
export async function layeredSinglePage(
  imageBuf: Buffer,
  mime: string,
  pageNum: number,
  rawFields: any[],
  mt: number,
  scVotes: number,
  extractionOptions: Record<string, any> = {}
): Promise<any> {
  if (!ENABLE_LAYERED_EXTRACTION) {
    // Fallback: direct JSON extraction
    const client = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL, timeout: 300_000 });
    const r = (await client.chat.completions.create({
      model: MODEL,
      max_tokens: mt,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:${mime};base64,${imageBuf.toString("base64")}` } },
            { type: "text", text: `Extract the following fields from this document as JSON: ${rawFields.map((f: any) => f.name).join(", ")}. The JSON keyword must appear.` },
          ],
        },
      ],
    })) as any;
    try {
      return { finalJson: JSON.parse(r.choices[0]?.message?.content || "{}"), stats: { totalApiCalls: 1, totalTokens: 0, elapsedMs: 0, page: pageNum } };
    } catch {
      return { finalJson: {}, stats: { totalApiCalls: 1, totalTokens: 0, elapsedMs: 0, page: pageNum } };
    }
  }

  // v8: Unified layered extraction
  const fieldSpecs: FieldSpec[] = rawFields.map((f: any) => ({
    name: f.name || "",
    labelPattern: f.label_pattern || f.labelPattern || f.name || "",
    positionHint: f.position_hint || f.positionHint || undefined,
    formatHint: f.format_hint || f.formatHint || undefined,
    example: f.example || undefined,
    allowedValues: f.allowed_values || f.allowedValues || undefined,
    contextRule: f.context_rule || f.contextRule || undefined,
    required: f.required === true,
  }));

  const config: LayeredExtractionConfig = {
    primaryModel: MODEL,
    ocrModel: OCR_MODEL,
    primaryBaseUrl: BASE_URL,
    ocrBaseUrl: OCR_BASE_URL || BASE_URL,
    apiKey: API_KEY!,
    enableThinking: ENABLE_THINKING,
    enableCE: true,
    enablePostProcess: true,
    forcedStrategy: (rawFields as any).strategy || undefined,
    ocrVerify: extractionOptions.ocr_verify !== false,
    costPolicy: extractionOptions.cost_policy,
    cachePolicy: extractionOptions.cache_policy,
    returnCostBreakdown: extractionOptions.return_cost_breakdown !== false,
    maxUnverifiedRequiredFields: typeof extractionOptions.max_unverified_required_fields === "number"
      ? extractionOptions.max_unverified_required_fields
      : undefined,
  } as any;

  const result = await layeredExtract(imageBuf, mime, fieldSpecs, config);

  // v8: Self-consistency is now handled inside layeredExtract L4
  // Only add extra votes if configured from caller
  if (ENABLE_SELF_CONSISTENCY && scVotes > CE_VOTES) {
    const scClient = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL, timeout: 300_000 });
    for (const fieldName of Object.keys(result.verifiedFields)) {
      const vf = result.verifiedFields[fieldName];
      if (vf.confidence === "low" && vf.value) {
        const fieldSpec = fieldSpecs.find((f) => f.name === fieldName);
        if (fieldSpec) {
          const hint = fieldSpec.formatHint ? `, format: ${fieldSpec.formatHint}` : "";
          const vote = await weightedConsistencyVote(
            scClient, MODEL, imageBuf.toString("base64"), mime,
            fieldName,
            `Extract ONLY the value for field "${fieldName}" (label: "${fieldSpec.labelPattern}"${hint}). Output just the value, nothing else.`,
            scVotes, 0.1
          );
          if (vote.agreement >= 0.66) {
            result.verifiedFields[fieldName] = { ...vf, value: vote.value, verified: true };
            result.finalJson[fieldName] = { value: vote.value, confidence: "high", verified: true };
          }
        }
      }
    }
  }

  result.stats = { ...result.stats, page: pageNum } as any;
  return result;
}
