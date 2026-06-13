import {
  CONCURRENCY,
  WEBP_QUALITY,
  WEBP_QUALITY_OCR,
  VL_HIGH_RES_ENABLED,
  SELF_CONSISTENCY_VOTES,
} from "../config/constants.js";

export type AccuracyMode = "fast" | "balanced" | "max";

export interface AccuracyPolicy {
  mode: AccuracyMode;
  imageQuality: number;
  concurrency: number;
  selfConsistencyVotes: number;
  vlHighResolutionImages: boolean;
  ocrVerify: boolean;
  strictValidation: boolean;
}

export function normalizeAccuracyMode(value: any): AccuracyMode {
  if (value === "fast" || value === "balanced" || value === "max") return value;
  return "balanced";
}

export function resolveAccuracyPolicy(args: Record<string, any> = {}): AccuracyPolicy {
  const mode = normalizeAccuracyMode(args.accuracy_mode);
  if (mode === "fast") {
    return {
      mode,
      imageQuality: Math.min(Number(args.image_quality) || WEBP_QUALITY, 85),
      concurrency: Number(args.concurrency) || CONCURRENCY,
      selfConsistencyVotes: Number(args.self_consistency_votes) || 1,
      vlHighResolutionImages: args.vl_high_resolution_images === true,
      ocrVerify: args.ocr_verify === true,
      strictValidation: false,
    };
  }
  if (mode === "max") {
    return {
      mode,
      imageQuality: Math.max(Number(args.image_quality) || WEBP_QUALITY_OCR, 95),
      concurrency: Math.max(1, Math.min(Number(args.concurrency) || CONCURRENCY, 6)),
      selfConsistencyVotes: Number(args.self_consistency_votes) || Math.max(SELF_CONSISTENCY_VOTES, 3),
      vlHighResolutionImages: args.vl_high_resolution_images !== false,
      ocrVerify: args.ocr_verify !== false,
      strictValidation: true,
    };
  }
  return {
    mode,
    imageQuality: Number(args.image_quality) || WEBP_QUALITY_OCR,
    concurrency: Number(args.concurrency) || CONCURRENCY,
    selfConsistencyVotes: Number(args.self_consistency_votes) || SELF_CONSISTENCY_VOTES,
    vlHighResolutionImages: args.vl_high_resolution_images ?? VL_HIGH_RES_ENABLED,
    ocrVerify: args.ocr_verify !== false,
    strictValidation: true,
  };
}
