/**
 * Vision-MCP v8.1: Enhanced image preprocessing pipeline for OCR
 * + Negative image detection and auto-inversion
 *
 * Four document-type strategies:
 * - scan: adaptive binarization + descreening + edge enhancement
 * - photo: CLAHE-like contrast + shadow removal + perspective correction
 * - table: preserve grayscale + light contrast (don't destroy gridlines)
 * - handwriting: aggressive enhancement + grayscale preservation
 * - negative: detect and invert white-on-black documents before standard processing
 *
 * v8.1 additions: detectInverted(), auto-negate in preprocessForOCR,
 *                 preprocessNegative preset mode
 *
 * All operations use sharp (libvips) which runs on CPU efficiently.
 */

import sharp from "sharp";
import type { PreprocessOptions, PreprocessResult, DocumentType } from "../config/types.js";
import {
  ENABLE_EDGE_ENHANCE, ENABLE_ADAPTIVE_THRESHOLD,
  ENABLE_PERSPECTIVE_CORRECT,
} from "../config/constants.js";

// ---------- Negative/Inverted image detection ----------

interface InvertDetection {
  isInverted: boolean;
  darkPixelRatio: number;
  median: number;
}

/**
 * Detect if an image is inverted (white text on black/dark background).
 * Common in microfilm scans, old photocopies, banking records.
 *
 * Detection:
 * 1. Downsample to max 400px for speed
 * 2. Count dark pixels (value < 128)
 * 3. If >60% pixels are dark AND median < 80 → inverted
 */
export async function detectInverted(buffer: Buffer): Promise<InvertDetection> {
  const meta = await sharp(buffer).metadata();

  const small = await sharp(buffer)
    .resize(Math.min(meta.width ?? 400, 400), undefined, { fit: "inside" })
    .greyscale()
    .raw()
    .toBuffer();

  let darkCount = 0;
  const total = small.length;
  const values = new Uint8Array(small);
  for (let i = 0; i < total; i++) {
    if (values[i] < 128) darkCount++;
  }
  const darkPixelRatio = darkCount / total;

  const sorted = new Uint8Array(values);
  sorted.sort();
  const median = sorted[Math.floor(total / 2)];

  const isInverted = darkPixelRatio > 0.6 && median < 80;

  console.error(
    `[detect-invert] isInverted=${isInverted} dark_ratio=${(darkPixelRatio * 100).toFixed(1)}% median=${median}`
  );

  return { isInverted, darkPixelRatio, median };
}

// ---------- Document type detection ----------

interface DocDetection {
  type: DocumentType;
  hasHandwriting: boolean;
  hasTableLines: boolean;
  contrastLevel: "low" | "normal" | "high";
  noiseLevel: "low" | "medium" | "high";
}

/**
 * Auto-detect document type based on histogram, edge density, and texture analysis.
 */
export async function detectDocumentType(buffer: Buffer): Promise<DocDetection> {
  const meta = await sharp(buffer).metadata();
  const stats = await sharp(buffer).stats();

  const channels: sharp.ChannelStats[] = stats.channels;
  const mainChannel = channels[0];

  const stdDev = mainChannel.stdev;
  const contrastLevel: DocDetection["contrastLevel"] =
    stdDev < 30 ? "low" : stdDev > 65 ? "high" : "normal";

  const small = await sharp(buffer).resize(100, 100, { fit: "fill" }).raw().toBuffer();
  let localVarSum = 0;
  for (let i = 32; i < small.length - 32; i++) {
    localVarSum += Math.abs(small[i] - small[i - 32]);
    localVarSum += Math.abs(small[i] - small[i + 1]);
  }
  const avgLocalVar = localVarSum / (small.length * 2);
  const noiseLevel: DocDetection["noiseLevel"] =
    avgLocalVar < 8 ? "low" : avgLocalVar > 20 ? "high" : "medium";

  const edgeBuf = await sharp(buffer)
    .resize(200, 200, { fit: "inside" })
    .greyscale()
    .convolve({
      width: 3, height: 3,
      kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    })
    .raw().toBuffer();
  let edgeDensity = 0;
  for (let i = 0; i < edgeBuf.length; i++) {
    if (edgeBuf[i] > 40) edgeDensity++;
  }
  const edgeRatio = edgeDensity / edgeBuf.length;
  const hasTableLines = edgeRatio > 0.04 && edgeRatio < 0.30 && noiseLevel !== "high";

  const hasHandwriting =
    noiseLevel === "high" &&
    contrastLevel !== "high" &&
    edgeRatio > 0.03 &&
    !hasTableLines;

  let type: DocumentType;
  if (hasTableLines) {
    type = "table";
  } else if (hasHandwriting) {
    type = "handwriting";
  } else if (contrastLevel === "low" && noiseLevel !== "low") {
    type = "photo";
  } else {
    type = "scan";
  }

  console.error(
    `[detect] type=${type} handwriting=${hasHandwriting} table=${hasTableLines} ` +
    `contrast=${contrastLevel} noise=${noiseLevel} edge_ratio=${edgeRatio.toFixed(4)}`
  );

  return { type, hasHandwriting, hasTableLines, contrastLevel, noiseLevel };
}

// ---------- Core preprocessing ----------

export interface PreprocessResultV8_1 extends PreprocessResult {
  wasInverted?: boolean;
}

export async function preprocessForOCR(
  inputBuffer: Buffer,
  options: PreprocessOptions = {}
): Promise<PreprocessResultV8_1> {
  const steps: string[] = [];
  let pipeline = sharp(inputBuffer);
  const meta = await pipeline.metadata();

  const opts: Required<PreprocessOptions> = {
    autoOrient: options.autoOrient ?? true,
    deskew: options.deskew ?? false,
    enhanceContrast: options.enhanceContrast ?? true,
    sharpen: options.sharpen ?? true,
    grayscale: options.grayscale ?? true,
    removeBackground: options.removeBackground ?? true,
    quality: options.quality ?? 95,
    docType: options.docType ?? "scan",
    edgeEnhance: options.edgeEnhance ?? ENABLE_EDGE_ENHANCE,
    adaptiveThreshold: options.adaptiveThreshold ?? ENABLE_ADAPTIVE_THRESHOLD,
    perspectiveCorrect: options.perspectiveCorrect ?? ENABLE_PERSPECTIVE_CORRECT,
    // v8.1
    detectNegative: (options as any).detectNegative ?? true,
    autoInvert: (options as any).autoInvert ?? true,
  };

  // v8.1: Detect and auto-invert negative (white-on-black) images FIRST
  let wasInverted = false;
  if ((opts as any).detectNegative) {
    const invResult = await detectInverted(inputBuffer);
    if (invResult.isInverted && (opts as any).autoInvert) {
      pipeline = pipeline.negate({ alpha: false });
      steps.push("invert(negative-detect)");
      wasInverted = true;
      console.error(
        `[preprocess] Negative image detected and inverted (dark_ratio=${(invResult.darkPixelRatio * 100).toFixed(1)}%)`
      );
    }
  }

  // Auto-detect document type if not specified
  let effectiveDocType = opts.docType;
  let detectedDocType: DocumentType | undefined;
  if (effectiveDocType === "mixed" as any || !effectiveDocType) {
    const detection = await detectDocumentType(inputBuffer);
    effectiveDocType = detection.type;
    detectedDocType = detection.type;
  }

  // If image was inverted, use lighter processing
  if (wasInverted) {
    effectiveDocType = "scan";
    (opts as any).adaptiveThreshold = false;
    (opts as any).edgeEnhance = false;
  }

  // Auto-orient based on EXIF
  if (opts.autoOrient) {
    pipeline = pipeline.rotate();
    steps.push("auto-orient");
  }

  // Grayscale for document processing (skip for handwriting to preserve nuance)
  if (opts.grayscale && (meta.channels ?? 3) >= 3 && effectiveDocType !== "handwriting") {
    pipeline = pipeline.greyscale();
    steps.push("grayscale");
  }

  // Document-type-specific processing
  if (effectiveDocType === "table") {
    if (opts.enhanceContrast) {
      pipeline = pipeline.linear(1.2, -15);  // v9: table contrast boost
      steps.push("contrast(table-light 1.2)");  // v9
    }
    if (opts.sharpen) {
      pipeline = pipeline.sharpen({ sigma: 0.5, m1: 0.5, m2: 0.2, x1: 1.5, y2: 6, y3: 10 });
      steps.push("sharpen(table sigma=0.5)");
    }
  } else if (effectiveDocType === "photo") {
    if (opts.removeBackground) {
      pipeline = pipeline.normalise();
      steps.push("normalize(shadow-removal)");
    }
    if (opts.enhanceContrast) {
      pipeline = pipeline.linear(1.4, -40);
      steps.push("contrast(photo 1.4,-40)");
    }
    if (opts.sharpen) {
      pipeline = pipeline.sharpen({ sigma: 0.9, m1: 0.9, m2: 0.3, x1: 2.0, y2: 8, y3: 12 });
      steps.push("sharpen(photo sigma=0.9)");
    }
  } else if (effectiveDocType === "handwriting") {
    if (opts.removeBackground) {
      pipeline = pipeline.normalise();
      steps.push("normalize");
    }
    if (opts.enhanceContrast) {
      pipeline = pipeline.linear(1.6, -50);
      steps.push("contrast(handwriting 1.6,-50)");
    }
    if (opts.sharpen) {
      pipeline = pipeline.sharpen({ sigma: 1.1, m1: 1.2, m2: 0.4, x1: 2.5, y2: 10, y3: 14 });
      steps.push("sharpen(handwriting sigma=1.1)");
    }
    if (opts.edgeEnhance) {
      pipeline = pipeline.convolve({
        width: 3, height: 3,
        kernel: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        scale: 0.3,
      });
      steps.push("sobel-edge(handwriting)");
    }
  } else {
    // scan: balanced approach
    if (opts.removeBackground) {
      pipeline = pipeline.normalise();
      steps.push("normalize");
    }
    if (opts.enhanceContrast) {
      pipeline = pipeline.linear(1.35, -35);  // v9: scan contrast boost
      steps.push("contrast(1.35,-35)");  // v9
    }
    if (opts.sharpen) {
      pipeline = pipeline.sharpen({ sigma: 0.7, m1: 0.7, m2: 0.25, x1: 1.8, y2: 7, y3: 10 });
      steps.push("sharpen(sigma=0.7)");  // v9
    }
    if (opts.edgeEnhance) {
      pipeline = pipeline.convolve({
        width: 3, height: 3,
        kernel: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        scale: 0.2,  // v9: reduced edge enhance
      });
      steps.push("sobel-edge(scan)");
    }
  }

  // v8: Adaptive threshold binarization (skip for inverted docs)
  if ((opts as any).adaptiveThreshold && effectiveDocType === "scan") {
    const buf = await pipeline.toBuffer();
    const thresh = await sharp(buf)
      .threshold(128)
      .blur(0.5)
      .toBuffer();
    pipeline = sharp(thresh);
    steps.push("adaptive-threshold");
  }

  const resultBuf = await pipeline.png({ quality: opts.quality }).toBuffer();
  const resultMeta = await sharp(resultBuf).metadata();

  console.error(
    `[preprocess] type=${effectiveDocType}${wasInverted ? " (was-inverted)" : ""} steps=[${steps.join(" -> ")}] | ` +
    `${meta.width}x${meta.height} -> ${resultMeta.width}x${resultMeta.height} | ` +
    `${(inputBuffer.length / 1024).toFixed(0)}KB -> ${(resultBuf.length / 1024).toFixed(0)}KB`
  );

  return {
    buffer: resultBuf,
    mime: "image/png",
    width: resultMeta.width ?? meta.width ?? 0,
    height: resultMeta.height ?? meta.height ?? 0,
    appliedSteps: steps,
    detectedDocType,
    wasInverted,
  };
}

// ---------- Preset modes ----------

/** Light mode: minimal processing, preserve original look */
export async function preprocessLight(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: false,
    removeBackground: false,
    enhanceContrast: false,
    sharpen: true,
    autoOrient: true,
    edgeEnhance: false,
    adaptiveThreshold: false,
    docType: "scan",
  });
}

/** Aggressive mode: maximum enhancement for OCR */
export async function preprocessAggressive(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: true,
    removeBackground: true,
    enhanceContrast: true,
    sharpen: true,
    autoOrient: true,
    edgeEnhance: true,
    docType: "handwriting",
  });
}

/** Table mode: preserve gridlines */
export async function preprocessTable(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: true,
    removeBackground: true,
    enhanceContrast: true,
    sharpen: false,
    autoOrient: true,
    edgeEnhance: false,
    adaptiveThreshold: false,
    docType: "table",
  });
}

/** v8: Handwriting-optimized mode */
export async function preprocessHandwriting(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: false,
    removeBackground: true,
    enhanceContrast: true,
    sharpen: true,
    autoOrient: true,
    edgeEnhance: true,
    adaptiveThreshold: false,
    docType: "handwriting",
  });
}

/** v8.1: Negative mode - detect and invert white-on-black documents */
export async function preprocessNegative(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: true,
    removeBackground: true,
    enhanceContrast: true,
    sharpen: true,
    autoOrient: true,
    edgeEnhance: false,
    adaptiveThreshold: false,
    detectNegative: true,
    autoInvert: true,
    docType: "scan",
  } as any);
}


/** v8.1: Scanned document optimized mode */
export async function preprocessScanned(inputBuffer: Buffer): Promise<PreprocessResult> {
  return preprocessForOCR(inputBuffer, {
    grayscale: false,
    removeBackground: true,
    enhanceContrast: true,
    sharpen: true,
    autoOrient: true,
    edgeEnhance: true,
    adaptiveThreshold: false,
    docType: "scan",
  });
}
