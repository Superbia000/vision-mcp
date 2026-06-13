/**
 * Vision-MCP v7: analyze_image tool
 */

import { readFileSync, statSync } from "fs";
import { extname } from "path";
import type { VisionProvider } from "../providers/base.js";
import { MAX_IMAGE_WIDTH, MAX_IMAGE_BYTES, WEBP_QUALITY, WEBP_QUALITY_OCR, ENABLE_PREPROCESSING, MIN_OCR_QUALITY , MODEL } from "../config/constants.js";
import { preprocessForOCR } from "../preprocessing/pipeline.js";
import { optimizeForVision } from "../rendering/image.js";
import { extToMime, isImageExt, isDocExtractionPrompt } from "../utils/helpers.js";
import { enhancePrompt } from "./helpers.js";

export async function handleAnalyzeImage(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const path = args.image_path as string;
  const prompt = args.prompt as string;
  const mt = (args.max_tokens as number) ?? 4096;
  const enableThinking = args.enable_thinking as boolean | undefined;
  const thinkingBudget = args.thinking_budget as number | undefined;
  const vlHighRes = args.vl_high_resolution_images as boolean | undefined;
  const maxPixels = args.max_pixels as number | undefined;
  const minPixels = args.min_pixels as number | undefined;
  const temperature = args.temperature as number | undefined;
  const topP = args.top_p as number | undefined;

  if (!path) return JSON.stringify({ error: "image_path required" });
  if (!prompt) return JSON.stringify({ error: "prompt required" });

  const ext = extname(path).toLowerCase();
  if (!isImageExt(ext)) return JSON.stringify({ error: `Unsupported format: ${ext}` });

  const info = statSync(path);
  if (info.size > MAX_IMAGE_BYTES) return JSON.stringify({ error: `Too large: ${(info.size / 1024 / 1024).toFixed(1)} MB` });

  const enhancedPrompt = enhancePrompt(prompt);
  const rawBuf = readFileSync(path);
  let finalBuf: Buffer = rawBuf;
  let finalMime = extToMime(ext);

  const isDoc = isDocExtractionPrompt(prompt);
  const imgMaxWidth = (args.max_image_width as number) ?? MAX_IMAGE_WIDTH;
  const imgQualityRaw = (args.image_quality as number) ?? (isDoc ? WEBP_QUALITY_OCR : WEBP_QUALITY);
  // v8.2: Enforce minimum quality for document prompts
  const imgQuality = (isDoc && imgQualityRaw < MIN_OCR_QUALITY) ? MIN_OCR_QUALITY : imgQualityRaw;
  const qualityAdjusted = isDoc && imgQualityRaw < MIN_OCR_QUALITY;

  // Preprocessing for document images
  let preprocessed = false;
  if (isDoc && ENABLE_PREPROCESSING) {
    try {
      const pp = await preprocessForOCR(rawBuf, {
        grayscale: true,
        removeBackground: true,
        enhanceContrast: true,
        sharpen: true,
      });
      finalBuf = Buffer.from(pp.buffer);
      finalMime = pp.mime;
      preprocessed = true;
    } catch (err: any) {
      console.error(`[analyze_image] Preprocessing skipped: ${err.message}`);
    }
  }

  // Image optimization (skip if already preprocessed)
  if (!preprocessed) {
    try {
      const meta = await import("sharp").then((m) => m.default(rawBuf).metadata());
      if (meta.width && meta.width > 0) {
        const img = { buffer: rawBuf, mime: finalMime, width: meta.width, height: meta.height || 0 };
        const opt = await optimizeForVision(img, { quality: imgQuality, maxWidth: imgMaxWidth });
        finalBuf = opt.buffer;
        finalMime = opt.mime;
      }
    } catch { /* skip optimization */ }
  }

  try {
    const r = await provider.chat({
      model: MODEL,
      max_tokens: mt,
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:${finalMime};base64,${finalBuf.toString("base64")}` } },
            { type: "text", text: enhancedPrompt },
          ],
        },
      ],
      enable_thinking: enableThinking,
      thinking_budget: thinkingBudget,
      vl_high_resolution_images: vlHighRes,
      max_pixels: maxPixels,
      min_pixels: minPixels,
      temperature,
      top_p: topP,
    });

    return JSON.stringify({
      success: true,
      ...(qualityAdjusted ? { quality_adjusted: true, adjusted_from: imgQualityRaw } : {}),
      text: r.text,
      reasoning: r.reasoning,
      input_tokens: r.it,
      output_tokens: r.ot,
    });
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

