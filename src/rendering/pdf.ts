/**
 * Vision-MCP v8.3: PDF rendering via mupdf WASM
 * v8.3: Image-based PDF detection + optimized native scale for each type.
 */

import * as mupdf from "mupdf";
import { createHash } from "crypto";
import { readFileSync, statSync } from "fs";
import type { ExtractedImage } from "../config/types.js";
import {
  MAX_PAGE_MEGAPIXELS,
  MAX_IMAGE_WIDTH,
  IMAGE_MAX_DIMENSION,
  ENABLE_PREPROCESSING, PREPROCESS_WITH_THINKING,
  RENDER_CACHE_ENABLED,
  RENDER_CACHE_MAX,
} from "../config/constants.js";
import { preprocessForOCR, preprocessAggressive } from "../preprocessing/pipeline.js";
import { LRUCache } from "../utils/helpers.js";

// Per-PDF optimal render scale cache (auto-detected from native image resolution)
const nativeScaleCache = new Map<string, number>();

// v8.3: Per-PDF type cache (image-based vs text-based)
const pdfTypeCache = new Map<string, "image" | "text">();

// Render cache keyed by file hash, page, render scale, preprocessing, and pixel controls.
const renderCache = new LRUCache<ExtractedImage>(RENDER_CACHE_MAX);
const fileHashCache = new Map<string, { sig: string; hash: string }>();

export interface RenderPageOptions {
  renderScale?: number;
  maxPixels?: number;
  losslessMode?: boolean;
}

function fileCacheId(path: string): string {
  const stat = statSync(path);
  const sig = `${stat.size}:${stat.mtimeMs}`;
  const cached = fileHashCache.get(path);
  if (cached?.sig === sig) return cached.hash;
  const hash = createHash("sha1").update(readFileSync(path)).digest("hex").slice(0, 16);
  fileHashCache.set(path, { sig, hash });
  return hash;
}

function cacheKey(path: string, pageNum: number, maxWidth: number, preprocess: boolean = false, skip: boolean = false, options: RenderPageOptions = {}): string {
  return [
    fileCacheId(path),
    pageNum,
    maxWidth,
    `pp${preprocess ? 1 : 0}`,
    `sk${skip ? 1 : 0}`,
    `rs${options.renderScale ?? "auto"}`,
    `mp${options.maxPixels ?? "auto"}`,
    `ll${options.losslessMode === false ? 0 : 1}`,
  ].join("::");
}

// v8.3: Export for external use (pdf-analyze needs it)
export function getPdfType(pdfPath: string): "image" | "text" | undefined {
  return pdfTypeCache.get(pdfPath);
}

// v8.3: Export native scale for external use
export function getNativeScale(pdfPath: string): number | undefined {
  return nativeScaleCache.get(pdfPath);
}

/**
 * v8.3: Detect whether a PDF is image-based (scanned) or text-based (vector).
 *
 * Image-based PDFs embed raster images at fixed native resolution.
 * Rendering below that loses detail, above that wastes tokens.
 *
 * Detection: render at scale=1 and scale=2, compare bytes-per-pixel.
 * If scale=1 already has high bpp (rich image detail), it's image-based.
 */
async function detectPdfType(
  page: any,
  pageW: number,
  pageH: number
): Promise<"image" | "text"> {
  // Render at scale=1
  const p1 = page.toPixmap(mupdf.Matrix.scale(1, 1), mupdf.ColorSpace.DeviceRGB);
  const w1 = p1.getWidth();
  const h1 = p1.getHeight();
  const png1 = Buffer.from(p1.asPNG());
  const bpp1 = png1.length / (w1 * h1);
  p1.destroy();

  // Render at scale=2
  const p2 = page.toPixmap(mupdf.Matrix.scale(2, 2), mupdf.ColorSpace.DeviceRGB);
  const w2 = p2.getWidth();
  const h2 = p2.getHeight();
  const png2 = Buffer.from(p2.asPNG());
  const bpp2 = png2.length / (w2 * h2);
  p2.destroy();

  // Image-based: scale=1 already has rich detail (bpp > 0.35)
  // AND bpp doesn't drop much between scale=1 and scale=2 (ratio > 0.7)
  const bppRatio = bpp2 / bpp1; const isImageBased = bppRatio > 0.8 || (bpp1 > 0.35 && bppRatio > 0.7);

  const type = isImageBased ? "image" : "text";
  console.error(
    `[pdf-type] ${type} (bpp@1x=${bpp1.toFixed(3)} bpp@2x=${bpp2.toFixed(3)} ratio=${(bpp2/bpp1).toFixed(3)})`
  );
  return type;
}

/**
 * v8.3: Auto-detect the native embedded image resolution.
 *
 * For image-based PDFs: probe at [1, 1.5, 2, 3, 4]x.
 *   The embedded image's native resolution is at scale=1.
 *   Higher scales just pixel-double without adding detail.
 *   Use the first scale where bpp plateaus (no meaningful detail gain).
 *
 * For text-based PDFs: probe at [2, 3, 4, 5, 6]x.
 *   Text requires higher scale for legibility.
 *   Use the scale where bpp stabilizes.
 *
 * Falls back to 4x render scale for text-based, 1x for image-based.
 */
async function detectNativeScale(
  page: any,
  pageW: number,
  pageH: number,
  pdfType: "image" | "text"
): Promise<number> {
  const probes = pdfType === "image"
    ? [1, 1.5, 2, 3, 4]
    : [2, 3, 4, 5, 6];

  const fallback = pdfType === "image" ? 1 : 4;
  let prevBpp = Infinity;
  let bestScale = fallback;

  for (const s of probes) {
    const mp = (pageW * s * pageH * s) / 1_000_000;
    if (mp > MAX_PAGE_MEGAPIXELS) break;

    const pixmap = page.toPixmap(
      mupdf.Matrix.scale(s, s),
      mupdf.ColorSpace.DeviceRGB
    );
    const w = pixmap.getWidth();
    const h = pixmap.getHeight();
    const png = Buffer.from(pixmap.asPNG());
    const bpp = png.length / (w * h);
    pixmap.destroy();

    // When bpp drops >25% from previous, we exceeded the useful scale
    if (prevBpp !== Infinity && bpp < prevBpp * 0.75) {
      bestScale = probes[probes.indexOf(s) - 1] || s;
      break;
    }
    prevBpp = bpp;
    bestScale = s;
  }

  return bestScale;
}

export async function renderPageSmart(
  pdfPath: string,
  pageNum: number,
  maxWidth: number,
  preprocessForOCR2: boolean = false,
  useAggressive: boolean = false,
  skipPreprocess: boolean = false,
  options: RenderPageOptions = {}
): Promise<ExtractedImage> {
  // Check cache
  if (RENDER_CACHE_ENABLED) {
    const key = cacheKey(pdfPath, pageNum, maxWidth, preprocessForOCR2, skipPreprocess, options);
    const cached = renderCache.get(key);
    if (cached) {
      console.error(`[render] Cache hit: page ${pageNum}`);
      return cached;
    }
  }

  const doc = await mupdf.Document.openDocument(pdfPath, "application/pdf");
  const total = doc.countPages();
  if (pageNum < 1 || pageNum > total) {
    doc.destroy();
    throw new Error(`Page ${pageNum} out of range (1-${total})`);
  }

  const page = doc.loadPage(pageNum - 1);
  const bounds = page.getBounds();
  const pageW = bounds[2] - bounds[0];
  const pageH = bounds[3] - bounds[1];

  // v8.3: Detect PDF type on first page (cached)
  let pdfType = pdfTypeCache.get(pdfPath);
  if (pdfType === undefined) {
    pdfType = await detectPdfType(page, pageW, pageH);
    pdfTypeCache.set(pdfPath, pdfType);
  }

  // v8.3: Auto-detect native scale (cached per PDF, uses pdfType)
  let optimalScale = nativeScaleCache.get(pdfPath);
  if (optimalScale === undefined) {
    optimalScale = await detectNativeScale(page, pageW, pageH, pdfType);
    nativeScaleCache.set(pdfPath, optimalScale);
    console.error(
      `[render] PDF type=${pdfType}, native scale=${optimalScale.toFixed(1)}x`
    );
  }

  // v8.3: Compute render scale based on PDF type
  let scale: number;
  if (options.renderScale && options.renderScale > 0 && pdfType !== "image") {
    scale = options.renderScale;
  } else if (pdfType === "image") {
    // Image-based: use native resolution, maxWidth acts as absolute cap only
    // Never downscale below 80% of native resolution for image-based PDFs
    const nativeWidth = pageW * optimalScale;
    if (maxWidth > 0 && maxWidth < nativeWidth) {
      // maxWidth is lower than native → downscale, but not below 80% native
      const minAcceptableWidth = nativeWidth * 0.8;
      const effectiveMax = Math.max(maxWidth, minAcceptableWidth);
      scale = effectiveMax / pageW;
    } else {
      scale = optimalScale;
    }
  } else {
    // Text-based: existing logic
    scale = maxWidth > 0 ? Math.min(maxWidth / pageW, optimalScale) : optimalScale;
  }

  // v8.3: Apply IMAGE_MAX_DIMENSION cap
  if (IMAGE_MAX_DIMENSION > 0) {
    const rawWidth = pageW * scale;
    const rawHeight = pageH * scale;
    if (rawWidth > IMAGE_MAX_DIMENSION || rawHeight > IMAGE_MAX_DIMENSION) {
      const aspectRatio = pageH / pageW;
      if (rawWidth >= rawHeight) {
        scale = IMAGE_MAX_DIMENSION / pageW;
      } else {
        scale = IMAGE_MAX_DIMENSION / pageH;
      }
    }
  }

  if (options.maxPixels && options.maxPixels > 0) {
    const rawPixels = pageW * scale * pageH * scale;
    if (rawPixels > options.maxPixels) {
      scale = Math.sqrt(options.maxPixels / (pageW * pageH));
    }
  }

  const mp = (pageW * scale * pageH * scale) / 1_000_000;
  if (mp > MAX_PAGE_MEGAPIXELS) {
    doc.destroy();
    throw new Error(`Page ${pageNum}: ${mp.toFixed(1)}MP exceeds limit (${MAX_PAGE_MEGAPIXELS}MP)`);
  }

  const pixmap = page.toPixmap(
    mupdf.Matrix.scale(scale, scale),
    mupdf.ColorSpace.DeviceRGB
  );
  const png = Buffer.from(pixmap.asPNG());
  const w = pixmap.getWidth();
  const h = pixmap.getHeight();
  pixmap.destroy();
  doc.destroy();

  let result: ExtractedImage = { buffer: png, mime: "image/png", width: w, height: h };

  // Apply preprocessing if requested (skip when thinking is ON unless explicitly allowed)
  if (preprocessForOCR2 && ENABLE_PREPROCESSING && (!skipPreprocess || PREPROCESS_WITH_THINKING)) {
    try {
      const pp = useAggressive
        ? await preprocessAggressive(png)
        : await preprocessForOCR(png, {
            grayscale: true,
            removeBackground: true,
            enhanceContrast: true,
            sharpen: true,
          });
      result = {
        buffer: Buffer.from(pp.buffer),
        mime: pp.mime,
        width: pp.width,
        height: pp.height,
      };
    } catch (err: any) {
      console.error(`[render] Preprocessing failed for page ${pageNum}: ${err.message}`);
    }
  }

  // Cache the result
  if (RENDER_CACHE_ENABLED) {
    renderCache.set(cacheKey(pdfPath, pageNum, maxWidth, preprocessForOCR2, skipPreprocess, options), result);
  }

  return result;
}

/**
 * Bridge: convert PDF page to image for image-based tools.
 */
export async function pdfPageToImage(
  pdfPath: string,
  pageNum: number = 1
): Promise<{ buffer: Buffer; mime: string; width: number; height: number }> {
  const doc = await mupdf.Document.openDocument(pdfPath, "application/pdf");
  const page = doc.loadPage(pageNum - 1);
  const bounds = page.getBounds();
  const pageW = bounds[2] - bounds[0];

  // v8.3: Use cached PDF type and native scale (same as renderPageSmart)
  let pdfType = pdfTypeCache.get(pdfPath);
  if (pdfType === undefined) {
    pdfType = await detectPdfType(page, pageW, bounds[3] - bounds[1]);
    pdfTypeCache.set(pdfPath, pdfType);
  }
  let optimalScale = nativeScaleCache.get(pdfPath);
  if (optimalScale === undefined) {
    optimalScale = await detectNativeScale(page, pageW, bounds[3] - bounds[1], pdfType);
    nativeScaleCache.set(pdfPath, optimalScale);
  }
  const scale = optimalScale;
  const pixmap = page.toPixmap(
    mupdf.Matrix.scale(scale, scale),
    mupdf.ColorSpace.DeviceRGB
  );
  const png = Buffer.from(pixmap.asPNG());
  const w = pixmap.getWidth();
  const h = pixmap.getHeight();
  pixmap.destroy();
  doc.destroy();
  console.error(
    `[render] PDF->PNG bridge: page ${pageNum} rendered ${w}x${h} (${(png.length / 1024).toFixed(0)}KB)`
  );
  return { buffer: png, mime: "image/png", width: w, height: h };
}

/** Get total page count for a PDF */
export async function getPdfPageCount(pdfPath: string): Promise<number> {
  let doc: any = null;
  try {
    doc = await mupdf.Document.openDocument(pdfPath, "application/pdf");
  } catch (err: any) {
    const errMsg = err.message || String(err);
    console.error(`[pdf] mupdf open failed for page count: ${errMsg}`);
    throw new Error(`Cannot open PDF: ${errMsg}`);
  }
  const total = doc.countPages();
  doc.destroy();
  return total;
}

/** Get detailed PDF info */
export async function getPdfInfo(pdfPath: string, fsize: number): Promise<{
  file_mb: number;
  total_pages: number;
  pages: { page: number; w_pt: number; h_pt: number }[];
}> {
  let doc: any = null;
  try {
    doc = await mupdf.Document.openDocument(pdfPath, "application/pdf");
  } catch (err: any) {
    const errMsg = err.message || String(err);
    console.error(`[pdf] mupdf open failed: ${errMsg}. Falling back to raw inspection.`);
    throw new Error(`PDF open failed: ${errMsg}. The file may be image-based, encrypted, or corrupted.`);
  }
  const total = doc.countPages();
  const pages: { page: number; w_pt: number; h_pt: number }[] = [];
  for (let i = 0; i < total; i++) {
    const p = doc.loadPage(i);
    const b = p.getBounds();
    pages.push({ page: i + 1, w_pt: Math.round(b[2] - b[0]), h_pt: Math.round(b[3] - b[1]) });
  }
  doc.destroy();
  return { file_mb: +(fsize / 1024 / 1024).toFixed(2), total_pages: total, pages };
}

/** Clear the render cache */
export function clearRenderCache(): void {
  renderCache.clear();
}

