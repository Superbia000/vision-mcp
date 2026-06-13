import {
  CONCURRENCY,
  CONCURRENCY_MAX,
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

function positiveNumber(value: any): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function resolveConcurrency(args: Record<string, any> = {}): number {
  const requested = positiveNumber(args.max_api_concurrency) ?? positiveNumber(args.concurrency) ?? CONCURRENCY;
  const max = positiveNumber(args.concurrency_max) ?? CONCURRENCY_MAX;
  return Math.max(1, Math.min(Math.floor(requested), Math.floor(max)));
}

export function resolveAccuracyPolicy(args: Record<string, any> = {}): AccuracyPolicy {
  const mode = normalizeAccuracyMode(args.accuracy_mode);
  const concurrency = resolveConcurrency(args);
  if (mode === "fast") {
    return {
      mode,
      imageQuality: Math.min(Number(args.image_quality) || WEBP_QUALITY, 85),
      concurrency,
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
      concurrency,
      selfConsistencyVotes: Number(args.self_consistency_votes) || Math.max(SELF_CONSISTENCY_VOTES, 3),
      vlHighResolutionImages: args.vl_high_resolution_images !== false,
      ocrVerify: args.ocr_verify !== false,
      strictValidation: true,
    };
  }
  return {
    mode,
    imageQuality: Number(args.image_quality) || WEBP_QUALITY_OCR,
    concurrency,
    selfConsistencyVotes: Number(args.self_consistency_votes) || SELF_CONSISTENCY_VOTES,
    vlHighResolutionImages: args.vl_high_resolution_images ?? VL_HIGH_RES_ENABLED,
    ocrVerify: args.ocr_verify !== false,
    strictValidation: true,
  };
}
