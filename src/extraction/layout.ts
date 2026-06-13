/**
 * Vision-MCP v8: L1 - Layout Analysis (enhanced)
 *
 * v8 improvements:
 * - Always vl_high_resolution_images for maximum bbox accuracy
 * - temperature=0 for deterministic layout
 * - Enhanced prompt with format hints and examples
 * - Qwen3-VL native 1000x1000 coordinate system
 */

import type { VisionProvider } from "../providers/base.js";
import type { DocumentLayout, FieldSpec } from "../config/types.js";
import { VL_HIGH_RES_ENABLED } from "../config/constants.js";

const LAYOUT_SYSTEM_PROMPT = `You are a precision document layout analysis expert. Your task is to identify and locate form fields in document images with maximum accuracy.

For each field, output:
1. The field label text as it appears on the document
2. The bounding box coordinates (normalized 0-1000 system, matching Qwen3-VL native resolution)
3. The raw text value next to/below the label
4. A confidence level: "high" (clearly readable), "medium" (somewhat unclear), "low" (blurry/handwritten/obscured)

Output as a JSON array of fields with this structure:
{
  "fields": [
    {
      "name": "field_name",
      "label": "text as printed on document",
      "value": "extracted value",
      "confidence": "high|medium|low",
      "bbox": {"x": 100, "y": 200, "w": 300, "h": 50}
    }
  ]
}

IMPORTANT:
- Use the 1000x1000 normalized coordinate system
- Include ALL visible fields even if unclear
- For unclear text, mark confidence as "low" and note what you can read
- Do not fabricate values for empty/missing fields - leave value as ""
- Extract values EXACTLY as shown, preserving original formatting
- For blurry characters, use [?] placeholder instead of guessing`;

export async function analyzeLayout(
  provider: VisionProvider,
  model: string,
  imageBase64: string,
  mime: string,
  fieldSpecs?: FieldSpec[]
): Promise<DocumentLayout> {
  const tStart = Date.now();

  const fieldHints = fieldSpecs
    ? `\n\nExpected fields to locate:\n${fieldSpecs
        .map(
          (f) =>
            `- "${f.name}": label like "${f.labelPattern}"` +
            `${f.positionHint ? ` (near ${f.positionHint})` : ""}` +
            `${f.formatHint ? ` [format: ${f.formatHint}]` : ""}` +
            `${f.example ? ` [example: ${f.example}]` : ""}` +
            `${f.allowedValues ? ` [allowed: ${f.allowedValues.join(", ")}]` : ""}`
        )
        .join("\n")}`
    : "";

  try {
    const r = await provider.chat({
      model,
      max_tokens: 4096,
      temperature: 0, // v8: deterministic output for layout
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:${mime};base64,${imageBase64}` } },
            {
              type: "text",
              text: `${LAYOUT_SYSTEM_PROMPT}\n\nAnalyze this document image. Identify all form fields, their labels, positions (1000x1000 coordinates), and values.${fieldHints}\n\nOutput as JSON.`,
            },
          ],
        },
      ],
      enable_thinking: false, // v8.3: always disable thinking for layout analysis
      vl_high_resolution_images: VL_HIGH_RES_ENABLED, // v8: always high-res for layout
      response_format: { type: "json_object" }, // v8: structured output
    });

    const content = r.text;
    const reasoning = r.reasoning;

    let layout: DocumentLayout;
    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      const jsonText = jsonMatch ? jsonMatch[0] : "";
      let parsed: any;
      try {
        parsed = jsonText ? JSON.parse(jsonText) : { fields: [] };
      } catch {
        // Some vision models corrupt bbox key names while keeping field names and values valid.
        // The current table route only needs values, so retry after dropping bbox objects.
        const withoutBbox = jsonText.replace(/,?\s*"bbox"\s*:\s*\{[^{}]*\}/g, "");
        parsed = withoutBbox ? JSON.parse(withoutBbox) : { fields: [] };
      }
      layout = {
        fields: (parsed.fields || []).map((f: any) => ({
          name: f.name || "",
          label: f.label || "",
          value: f.value || "",
          confidence: f.confidence || "medium",
          bbox: f.bbox,
        })),
        rawText: content,
        reasoning: reasoning || undefined,
      };
      (layout as any).usage = { input_tokens: r.it, output_tokens: r.ot };
    } catch {
      layout = { fields: [], rawText: content, reasoning: reasoning || undefined };
      (layout as any).usage = { input_tokens: r.it, output_tokens: r.ot };
    }

    console.error(
      `[L1] Layout analysis: ${layout.fields.length} fields in ${Date.now() - tStart}ms`
    );
    return layout;
  } catch (err: any) {
    console.error(`[L1] Layout analysis failed: ${err.message}`);
    return {
      fields: fieldSpecs
        ? fieldSpecs.map((f) => ({
            name: f.name,
            label: f.labelPattern,
            value: "",
            confidence: "low" as const,
          }))
        : [],
      rawText: "",
    };
  }
}
