/**
 * Vision-MCP v7: Shared extraction helpers for verification tools
 * (Re-exports from post-processing module for backward compat)
 */

import OpenAI from "openai";
import { MODEL, FIX_MODEL, THINKING_BUDGET } from "../config/constants.js";
import type { FieldValidationRule, ValidationError } from "../config/types.js";

/** Thinking + Structured Output two-stage workflow */
export async function thinkingThenStructure(
  client: OpenAI,
  imageBuf: Buffer,
  mime: string,
  prompt: string
): Promise<{ reasoning: string; json: Record<string, any> }> {
  console.error("[thinking-structured] Phase 1: Thinking mode...");

  // Phase 1: Thinking mode
  const thinkParams: any = {
    model: MODEL,
    max_tokens: 8192,
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${mime};base64,${imageBuf.toString("base64")}` } },
        { type: "text", text: `${prompt}\n\nThink step by step about the document content, layout, and each field value. Then output the result.` },
      ],
    }],
    enable_thinking: true,
    stream: true,
  };
  if (THINKING_BUDGET) thinkParams.thinking_budget = THINKING_BUDGET;

  let fullContent = "";
  const stream = await client.chat.completions.create(thinkParams as any);
  for await (const chunk of stream as any) {
    const delta = chunk.choices?.[0]?.delta;
    if (delta?.content) fullContent += delta.content;
  }

  // Phase 2: Multi-strategy JSON extraction
  const strategies = [
    () => { const m = fullContent.match(/\{[\s\S]*\}/); return m ? JSON.parse(m[0]) : null; },
    () => { const cleaned = fullContent.replace(/```json\s*|```\s*/g, ""); const m = cleaned.match(/\{[\s\S]*\}/); return m ? JSON.parse(m[0]) : null; },
    () => JSON.parse(fullContent),
  ];

  for (const strat of strategies) {
    try {
      const parsed = strat();
      if (parsed && typeof parsed === "object" && Object.keys(parsed).length > 0) {
        return { reasoning: fullContent, json: parsed };
      }
    } catch { /* try next */ }
  }

  // Phase 3: Fix with Flash model
  console.error("[thinking-structured] Phase 2: JSON format fixing...");
  const fixR = (await client.chat.completions.create({
    model: FIX_MODEL,
    max_tokens: 4096,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a JSON format expert. Fix the following text into valid JSON. The JSON keyword must appear." },
      { role: "user", content: fullContent },
    ],
  })) as any;

  try {
    return { reasoning: fullContent, json: JSON.parse(fixR.choices[0]?.message?.content || "{}") };
  } catch {
    return { reasoning: fullContent, json: { raw: fullContent } };
  }
}

/** Validate fields against rules */
export function validateFields(
  values: Record<string, string>,
  rules: FieldValidationRule[]
): ValidationError[] {
  const errors: ValidationError[] = [];
  for (const rule of rules) {
    const value = values[rule.field] || "";
    if (rule.pattern) {
      try {
        const regex = new RegExp(rule.pattern);
        if (!regex.test(value)) {
          errors.push({ field: rule.field, value, rule: `pattern: ${rule.pattern}`, message: `"${value}" does not match expected pattern` });
        }
      } catch { /* invalid regex */ }
    }
    if (rule.minLength !== undefined && value.length < rule.minLength) {
      errors.push({ field: rule.field, value, rule: `minLength: ${rule.minLength}`, message: `"${value}" too short (${value.length} < ${rule.minLength})` });
    }
    if (rule.maxLength !== undefined && value.length > rule.maxLength) {
      errors.push({ field: rule.field, value, rule: `maxLength: ${rule.maxLength}`, message: `"${value}" too long (${value.length} > ${rule.maxLength})` });
    }
    if (rule.allowedValues && !rule.allowedValues.includes(value)) {
      errors.push({ field: rule.field, value, rule: `allowedValues: [${rule.allowedValues.join(", ")}]`, message: `"${value}" not allowed` });
    }
  }
  return errors;
}

/** Retry with error context */
export async function retryWithErrorContext(
  client: OpenAI,
  imageBuf: Buffer,
  mime: string,
  prompt: string,
  previousError: string,
  previousOutput?: string
): Promise<string> {
  const errorPrompt = `${prompt}\n\nYour previous attempt had the following error:\n${previousError}\n${previousOutput ? `\nPrevious output:\n${previousOutput}` : ""}\n\nPlease fix the error and provide the correct extraction. Return valid JSON.`;

  const r = (await client.chat.completions.create({
    model: MODEL,
    max_tokens: 4096,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [{
      role: "user",
      content: [
        { type: "image_url", image_url: { url: `data:${mime};base64,${imageBuf.toString("base64")}` } },
        { type: "text", text: errorPrompt },
      ],
    }],
  })) as any;

  return r.choices[0]?.message?.content || "{}";
}
