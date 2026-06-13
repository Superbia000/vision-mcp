/**
 * Vision-MCP v8.3: Image optimization via sharp
 * v8.3: PNG lossless-first with smart fallback compression.
 * Image-based PDFs: keep PNG (no generational loss).
 * Only compress when file size exceeds API limits or token budget overflows.
 */

import sharp from "sharp";
import type { ExtractedImage, OptimizedImage } from "../config/types.js";
import {
  MAX_IMAGE_WIDTH,
  IMAGE_PDF_LOSSLESS,
  IMAGE_MAX_DIMENSION,
  IMAGE_MAX_BYTES,
  IMAGE_COMPRESSION_FALLBACK,
  IMAGE_JPEG_QUALITY,
  LOSSLESS_MODE,
  WEBP_QUALITY,
} from "../config/constants.js";

export interface OptimizeOptions {
  quality?: number;
  maxWidth?: number;
  preferLossless?: boolean;
  maxBytes?: number;
  maxTokens?: number;
  isImageBasedPdf?: boolean;
}

/**
 * v8.3: Optimize image for vision API submission.
 *
 * Strategy:
 * - Image-based PDFs: keep PNG lossless unless exceeds size/token limits
 * - Text-based PDFs / regular images: WebP q90 (existing behavior)
 * - When compression needed: JPEG q92 (better universality across models)
 * - Never upscale (withoutEnlargement always true)
 * - Cap max dimension at IMAGE_MAX_DIMENSION
 */
export async function optimizeForVision(
  img: ExtractedImage,
  qualityOrOptions: number | OptimizeOptions = WEBP_QUALITY,
  maxWidth: number = MAX_IMAGE_WIDTH
): Promise<OptimizedImage> {
  // v8.3: support both legacy (quality, maxWidth) and new (options object) signatures
  const opts: OptimizeOptions = typeof qualityOrOptions === "number"
    ? { quality: qualityOrOptions, maxWidth }
    : qualityOrOptions;

  const {
    quality = WEBP_QUALITY,
    maxWidth: optMaxWidth = MAX_IMAGE_WIDTH,
    preferLossless = false,
    maxBytes = IMAGE_MAX_BYTES,
    isImageBasedPdf = false,
  } = opts;

  // Step 1: Compute target width with max dimension cap (v8.3)
  let targetW = optMaxWidth > 0 ? Math.min(img.width, optMaxWidth) : img.width;
  if (IMAGE_MAX_DIMENSION > 0) {
    targetW = Math.min(targetW, IMAGE_MAX_DIMENSION);
    // Also check height
    const aspectRatio = img.height / img.width;
    const targetH = Math.round(targetW * aspectRatio);
    if (targetH > IMAGE_MAX_DIMENSION) {
      targetW = Math.round(IMAGE_MAX_DIMENSION / aspectRatio);
    }
  }

  const needsResize = targetW < img.width;

  // Step 2: Decide format strategy (v8.3)
  const shouldUseLossless =
    LOSSLESS_MODE || (isImageBasedPdf && IMAGE_PDF_LOSSLESS) || preferLossless;

  if (shouldUseLossless) {
    // PNG lossless path
    let pipeline = sharp(img.buffer);
    if (needsResize) {
      pipeline = pipeline.resize({ width: targetW, withoutEnlargement: true });
    }
    const pngBuf = await pipeline.png().toBuffer();
    const pngSizeMB = pngBuf.length / (1024 * 1024);

    console.error(
      `[image] PNG lossless: ${(img.buffer.length / 1024).toFixed(0)}KB -> ${(pngBuf.length / 1024).toFixed(0)}KB ` +
      `(${targetW}x${Math.round(targetW * (img.height / img.width))})`
    );
    if (pngSizeMB >= maxBytes / (1024 * 1024)) {
      console.error(`[image] PNG exceeds ${(maxBytes / 1024 / 1024).toFixed(1)}MB limit; keeping PNG because lossless_mode is enabled`);
    }
    return { buffer: pngBuf, mime: "image/png" };
  }

  // Step 3: Compression path (legacy WebP or JPEG fallback)
  let pipeline = sharp(img.buffer);
  if (needsResize) {
    pipeline = pipeline.resize({ width: targetW, withoutEnlargement: true });
  }

  if (IMAGE_COMPRESSION_FALLBACK === "jpeg") {
    // JPEG: better universal compatibility
    const jpgBuf = await pipeline.jpeg({ quality: IMAGE_JPEG_QUALITY }).toBuffer();
    const ratio = ((jpgBuf.length / img.buffer.length) * 100).toFixed(0);
    console.error(
      `[image] JPEG q${IMAGE_JPEG_QUALITY}: ${(img.buffer.length / 1024).toFixed(0)}KB -> ${(jpgBuf.length / 1024).toFixed(0)}KB (${ratio}%)`
    );
    return { buffer: jpgBuf, mime: "image/jpeg" };
  } else {
    // WebP (existing behavior)
    const webpBuf = await pipeline.webp({ quality }).toBuffer();
    const ratio = ((webpBuf.length / img.buffer.length) * 100).toFixed(0);
    console.error(
      `[image] WebP q${quality}: ${(img.buffer.length / 1024).toFixed(0)}KB -> ${(webpBuf.length / 1024).toFixed(0)}KB (${ratio}%)`
    );
    return { buffer: webpBuf, mime: "image/webp" };
  }
}

/**
 * Read image metadata without loading full buffer.
 */
export async function getImageMetadata(
  buffer: Buffer
): Promise<{ width: number; height: number; format?: string }> {
  const meta = await sharp(buffer).metadata();
  return {
    width: meta.width || 0,
    height: meta.height || 0,
    format: meta.format,
  };
}

/**
 * Convert any supported image format to PNG buffer.
 */
export async function toPngBuffer(buffer: Buffer): Promise<Buffer> {
  return sharp(buffer).png().toBuffer();
}

/**
 * Get optimal quality setting based on document type.
 */
export function getOptimalQuality(
  isDocument: boolean,
  baseQuality: number,
  ocrQuality: number
): number {
  return isDocument ? ocrQuality : baseQuality;
}
