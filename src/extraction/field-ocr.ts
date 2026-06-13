/**
 * Vision-MCP v8: L2 - Per-Field OCR Extraction (enhanced)
 *
 * v8 improvements:
 * - Dedicated OCR model (qwen-vl-ocr-2025-11-20) with vl_high_resolution_images
 * - temperature=0 for deterministic extraction
 * - response_format: json_object for structured output
 * - Enhanced prompt with format hints, examples, and allowed values
 * - 25% padding for context in field cropping
 */

import sharp from "sharp";
import type { VisionProvider } from "../providers/base.js";
import type { FieldSpec, LocatedField } from "../config/types.js";
import { preprocessScanned } from "../preprocessing/pipeline.js";
import { VL_HIGH_RES_ENABLED, ENABLE_STRUCTURED_OUTPUT, L2_CROP_PADDING, L2_PREPROCESS_ENABLED } from "../config/constants.js";
import { estimateImageTokens } from "../runtime/cost.js";

// ---------- Field cropping ----------

export async function cropField(
  imageBuffer: Buffer,
  bbox: { x: number; y: number; w: number; h: number },
  imageWidth: number,
  imageHeight: number
): Promise<{ buffer: Buffer; mime: string }> {
  const scaleX = imageWidth / 1000;
  const scaleY = imageHeight / 1000;

  const left = Math.max(0, Math.floor(bbox.x * scaleX));
  const top = Math.max(0, Math.floor(bbox.y * scaleY));
  const width = Math.min(imageWidth - left, Math.ceil(bbox.w * scaleX));
  const height = Math.min(imageHeight - top, Math.ceil(bbox.h * scaleY));

  // Add padding (configurable via VISION_L2_PADDING, default 40% for better context in OCR)
  const padRatio = L2_CROP_PADDING;
  const padX = Math.floor(width * padRatio);
  const padY = Math.floor(height * padRatio);
  const cropLeft = Math.max(0, left - padX);
  const cropTop = Math.max(0, top - padY);
  const cropWidth = Math.min(imageWidth - cropLeft, width + padX * 2);
  const cropHeight = Math.min(imageHeight - cropTop, height + padY * 2);

  const cropped = await sharp(imageBuffer)
    .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
    .png()
    .toBuffer();

  return { buffer: cropped, mime: "image/png" };
}

/** v8.1: Crop field with preprocessing for scanned documents */
export async function cropFieldPreprocessed(
  imageBuffer: Buffer,
  bbox: { x: number; y: number; w: number; h: number },
  imageWidth: number,
  imageHeight: number
): Promise<{ buffer: Buffer; mime: string }> {
  const cropped = await cropField(imageBuffer, bbox, imageWidth, imageHeight);
  if (L2_PREPROCESS_ENABLED) {
    const pp = await preprocessScanned(cropped.buffer);
    return { buffer: Buffer.from(pp.buffer), mime: pp.mime };
  }
  return cropped;
}

// ---------- Field value extraction ----------

const FIELD_EXTRACT_PROMPT = `You are a precision text extraction specialist. Extract the exact text value from this document field image.
Rules:
1. Output ONLY the extracted value in JSON: {"value": "..."}
2. Preserve original formatting (dates, numbers, special characters)
3. For unclear characters, use [?] placeholder
4. If the field is empty, output {"value": ""}
5. Do NOT output the field label text - output only the data VALUE
6. For handwritten text, trace character by character from left to right
7. Do not fabricate or guess - if unsure, use [?]`;

export async function extractFieldValue(
  provider: VisionProvider,
  ocrModel: string,
  fieldImageBase64: string,
  fieldSpec: FieldSpec,
  mime?: string,
  fieldImageBuffer?: Buffer
): Promise<{ value: string; confidence: string; input_tokens?: number; output_tokens?: number; estimated_image_tokens?: number }> {
  let formatConstraint = "";
  if (fieldSpec.formatHint) {
    formatConstraint += `\nExpected format: ${fieldSpec.formatHint}`;
  }
  if (fieldSpec.example) {
    formatConstraint += `\nExample value: ${fieldSpec.example}`;
  }
  if (fieldSpec.allowedValues && fieldSpec.allowedValues.length > 0) {
    formatConstraint += `\nAllowed values: ${fieldSpec.allowedValues.join(", ")}`;
  }

  try {
    const estimatedImageTokens = fieldImageBuffer ? await estimateImageTokens(fieldImageBuffer) : undefined;
    const r = await provider.chat({
      model: ocrModel,
      max_tokens: 512,
      temperature: 0, // v8: deterministic extraction
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: `data:${mime || "image/png"};base64,${fieldImageBase64}`,
              },
            },
            {
              type: "text",
              text: `${FIELD_EXTRACT_PROMPT}\n\nField name: "${fieldSpec.name}"\nLabel pattern: "${fieldSpec.labelPattern}"${formatConstraint}\n\nExtract the exact value for this field. Output as JSON: {"value": "..."}`,
            },
          ],
        },
      ],
      vl_high_resolution_images: VL_HIGH_RES_ENABLED,
      enable_thinking: false, // v8.3: no thinking for direct OCR reads
      ...(ENABLE_STRUCTURED_OUTPUT ? { response_format: { type: "json_object" } } : {}),
    });

    const rawText = (r.text || "").trim();
    let value = rawText;

    // Parse JSON response
    try {
      const parsed = JSON.parse(rawText);
      value = parsed.value ?? rawText;
    } catch {
      // Fall back to raw text, strip quotes
      value = rawText.replace(/^["']|["']$/g, "");
    }

    const hasUncertainty = value.includes("[?]") || !value;

    return {
      value,
      confidence: hasUncertainty ? "low" : "high",
      input_tokens: r.it,
      output_tokens: r.ot,
      estimated_image_tokens: estimatedImageTokens,
    };
  } catch (err: any) {
    console.error(`[L2] Field "${fieldSpec.name}" extraction failed: ${err.message}`);
    return { value: "", confidence: "low" };
  }
}

// ---------- Parallel field extraction (all fields at once) ----------

export interface FieldExtractionTask {
  spec: FieldSpec;
  layoutField?: LocatedField;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
}

export async function extractFieldsParallel(
  provider: VisionProvider,
  ocrModel: string,
  tasks: FieldExtractionTask[],
  concurrency: number = 10
): Promise<Map<string, { value: string; confidence: string }>> {
  const results = new Map<string, { value: string; confidence: string }>();

  // v8.2: True parallel execution - all fields fire at once
  const allResults = await Promise.all(
    tasks.map(async (task) => {
      const { spec, layoutField, imageBuffer, imageWidth, imageHeight } = task;
      try {
        if (layoutField && layoutField.bbox) {
          const cropped = await cropFieldPreprocessed(imageBuffer, layoutField.bbox, imageWidth, imageHeight);
          const fieldB64 = cropped.buffer.toString("base64");
          const result = await extractFieldValue(provider, ocrModel, fieldB64, spec, cropped.mime, cropped.buffer);
          return { name: spec.name, result, apiCall: true };
        } else {
          const layoutValue = layoutField ? layoutField.value || "" : "";
          return { name: spec.name, result: { value: layoutValue, confidence: layoutField ? layoutField.confidence || "low" : "low" }, apiCall: false };
        }
      } catch (err) {
        return { name: spec.name, result: { value: "", confidence: "low" }, apiCall: false };
      }
    })
  );

  for (const r of allResults) {
    results.set(r.name, r.result);
  }
  return results;
}
