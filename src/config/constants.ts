/**
 * Vision-MCP v10: Centralized configuration from environment variables.
 * Keeps the existing OCR/PDF quality switches while the public MCP surface is
 * consolidated into five categorized tools.
 */

export const API_KEY = process.env.VISION_API_KEY || process.env.KIMI_API_KEY || "";

export const BASE_URL = process.env.VISION_BASE_URL || process.env.KIMI_BASE_URL || "https://dashscope.aliyuncs.com/compatible-mode/v1";

export const MODEL = process.env.VISION_MODEL || process.env.KIMI_MODEL || "qwen3-vl-plus";

// Detect provider type: qwen (DashScope) or openai-compat
export const IS_QWEN = BASE_URL.includes("dashscope") || BASE_URL.includes("aliyuncs");
export const PROVIDER_TYPE: "qwen" | "openai" = IS_QWEN ? "qwen" : "openai";

// Rendering / Size limits
export const DEFAULT_SCALE = 1.0;
export const MAX_OUTPUT_TOKENS = (() => { const n = Number(process.env.VISION_MAX_TOKENS || process.env.KIMI_MAX_TOKENS); return isNaN(n) || n <= 0 ? 16384 : n; })();
export const MAX_PAGE_MEGAPIXELS = 25;
export const REQUEST_TIMEOUT_MS = 300_000;
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const MAX_VIDEO_BYTES_LOCAL = 100 * 1024 * 1024;
export const MAX_VIDEO_BYTES_URL = 2 * 1024 * 1024 * 1024;
export const MAX_VIDEO_DURATION_SEC = Number(process.env.VISION_VIDEO_MAX_DURATION_SEC) || 3600;

export const SUPPORTED_IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);
export const SUPPORTED_VIDEO_EXTS = new Set([
  ".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv", ".3gpp",
]);

// Concurrency / Optimization
export const CONCURRENCY = (() => { const n = Number(process.env.VISION_CONCURRENCY); return isNaN(n) ? 10 : n; })();
export const CONCURRENCY_MAX = Number(process.env.VISION_CONCURRENCY_MAX) || 50;
export const ADAPTIVE_CONCURRENCY = process.env.VISION_ADAPTIVE_CONCURRENCY !== "false";
export const WEBP_QUALITY = Number(process.env.VISION_WEBP_QUALITY) || 90;
export const MAX_IMAGE_WIDTH = (() => { const n = Number(process.env.VISION_MAX_IMAGE_WIDTH); return isNaN(n) ? 2048 : n; })();
export const RETRY_MAX = Number(process.env.VISION_RETRY_MAX) || 3;
export const RETRY_BASE_MS = 1000;
export const PDF_CHUNK_SIZE = Number(process.env.VISION_PDF_CHUNK_SIZE) || 20;
export const PDF_STRATEGY = process.env.VISION_PDF_STRATEGY || "auto";
export const BATCH_MAX_PAGES = Number(process.env.VISION_BATCH_MAX_PAGES) || 50;
export const PDF_SOFT_CAP = 200;

// Video
export const VIDEO_FPS_DEFAULT = Number(process.env.VISION_VIDEO_FPS) || 1.0;
export const VIDEO_CHUNK_DURATION_SEC = Number(process.env.VISION_VIDEO_CHUNK_DURATION_SEC) || 1800;
export const VIDEO_MAX_FRAMES = 2000;
export const WEBP_QUALITY_OCR = Number(process.env.VISION_WEBP_QUALITY_OCR) || 95;

// Thinking mode (Qwen)
export const ENABLE_THINKING_DEFAULT = false;
export const ENABLE_THINKING = (() => {
  const v = process.env.VISION_ENABLE_THINKING;
  if (v === "true") return true;
  if (v === "false") return false;
  return ENABLE_THINKING_DEFAULT;
})();
export const THINKING_BUDGET = (() => { const n = Number(process.env.VISION_THINKING_BUDGET); return isNaN(n) || n < 0 ? 4096 : n; })();

// Temperature & Top-P
export const TEMPERATURE = process.env.VISION_TEMPERATURE ? Number(process.env.VISION_TEMPERATURE) : undefined;
export const TOP_P = process.env.VISION_TOP_P ? Number(process.env.VISION_TOP_P) : undefined;

// Token budget
export const TOKEN_BUDGET_ENABLED = process.env.VISION_TOKEN_BUDGET_ENABLED === "true";
export const TOKEN_BUDGET_K = Number(process.env.VISION_TOKEN_BUDGET_K) || 32;

// OCR & Extraction
export const OCR_MODEL = process.env.VISION_OCR_MODEL || "qwen-vl-ocr-2025-11-20";
export const OCR_BASE_URL = process.env.VISION_OCR_BASE_URL || "";
export const ENABLE_STRUCTURED_OUTPUT = process.env.VISION_STRUCTURED_OUTPUT !== "false";
export const FIX_MODEL = process.env.VISION_FIX_MODEL || "qwen3-vl-flash";
export const SELF_CONSISTENCY_VOTES = Number(process.env.VISION_SELF_CONSISTENCY_VOTES) || 2;  // v9: reduced from 3
export const COST_POLICY =
  (process.env.VISION_COST_POLICY as "quality_first" | "prefer_batch" | "realtime_only") || "quality_first";
export const CACHE_POLICY =
  (process.env.VISION_CACHE_POLICY as "auto" | "off" | "explicit") || "auto";
export const RETURN_COST_BREAKDOWN = process.env.VISION_RETURN_COST_BREAKDOWN !== "false";
export const MAX_UNVERIFIED_REQUIRED_FIELDS = (() => {
  const n = Number(process.env.VISION_MAX_UNVERIFIED_REQUIRED_FIELDS);
  return isNaN(n) || n < 0 ? 0 : n;
})();

// Feature switches (all default ON)
export const ENABLE_PREPROCESSING = process.env.VISION_ENABLE_PREPROCESSING !== "false";
export const ENABLE_LAYERED_EXTRACTION = process.env.VISION_ENABLE_LAYERED_EXTRACTION !== "false";
export const ENABLE_SELF_CONSISTENCY = process.env.VISION_ENABLE_SELF_CONSISTENCY !== "false";

// v7: Render cache
export const RENDER_CACHE_ENABLED = process.env.VISION_RENDER_CACHE !== "false";
export const RENDER_CACHE_MAX = Number(process.env.VISION_RENDER_CACHE_MAX) || 200;

// v7: Streaming output
export const STREAM_RESULTS = process.env.VISION_STREAM_RESULTS === "true";

// ---- v8: Resolution Control ----

/** When true, L1 layout and L2 OCR use vl_high_resolution_images (16384*32*32 pixels) */
export const VL_HIGH_RES_ENABLED = process.env.VISION_VL_HIGH_RES !== "false";

/** Default max_pixels for text documents (32*32*8192 = 8.3M) */
export const MAX_PIXELS_TEXT = (() => { const n = Number(process.env.VISION_MAX_PIXELS_TEXT); return isNaN(n) ? 32 * 32 * 8192 : n; })();

/** Default max_pixels for small-text/dense documents (32*32*16384 = 16.7M) */
export const MAX_PIXELS_DENSE = (() => { const n = Number(process.env.VISION_MAX_PIXELS_DENSE); return isNaN(n) ? 32 * 32 * 16384 : n; })();

/** Default min_pixels (4*32*32 = 4096) */
export const MIN_PIXELS = (() => { const n = Number(process.env.VISION_MIN_PIXELS); return isNaN(n) ? 4 * 32 * 32 : n; })();

/** Minimum PDF render DPI */
export const MIN_PDF_DPI = Number(process.env.VISION_MIN_PDF_DPI) || 300;

// ---- v8: Consensus Entropy (CE) ----

/** Enable consensus entropy scoring */
export const ENABLE_CE = process.env.VISION_ENABLE_CE !== "false";

/** CE threshold above which fields are flagged for review */
export const CE_THRESHOLD = Number(process.env.VISION_CE_THRESHOLD) || 0.3;  // v9: lowered from 0.5

/** Number of self-consistency votes for CE calculation */
export const CE_VOTES = Number(process.env.VISION_CE_VOTES) || 3;

/** Temperature for CE diversity sampling */
export const CE_TEMPERATURE = Number(process.env.VISION_CE_TEMPERATURE) || 0.4;

// ---- v8: Post-processing ----

/** Enable format-aware OCR error correction */
export const ENABLE_POST_PROCESS = process.env.VISION_ENABLE_POST_PROCESS !== "false";

/** Enable cross-field context validation */
export const ENABLE_CROSS_FIELD = process.env.VISION_ENABLE_CROSS_FIELD !== "false";

// ---- v8: Preprocessing Enhancements ----

/** Enable Sobel edge enhancement in preprocessing */
export const ENABLE_EDGE_ENHANCE = process.env.VISION_ENABLE_EDGE_ENHANCE !== "false";

/** Enable adaptive threshold binarization for scanned documents */
export const ENABLE_ADAPTIVE_THRESHOLD = process.env.VISION_ENABLE_ADAPTIVE_THRESHOLD === "true";

/** Enable perspective correction for photo documents */
export const ENABLE_PERSPECTIVE_CORRECT = process.env.VISION_ENABLE_PERSPECTIVE_CORRECT !== "false";

// ---- v8: Handwriting ----

/** Default language hint for handwriting */
export const HANDWRITING_LANG = process.env.VISION_HANDWRITING_LANG || "";

// ---- v8.1: Extraction Strategy Router ----

/** Force extraction strategy: auto | full-page | layered */
export const EXTRACTION_STRATEGY = process.env.VISION_EXTRACTION_STRATEGY || "auto";

/** Strategy to use for scanned/handwriting documents when auto-routing */
export const SCANNED_STRATEGY = process.env.VISION_SCANNED_STRATEGY || "full-page";

/** Enable multi-pass consistency voting for field extraction */
export const MULTIPASS_ENABLED = process.env.VISION_MULTIPASS_ENABLED !== "false";

/** Number of multi-pass voting rounds (default 3) */
export const MULTIPASS_VOTES = (() => { const n = Number(process.env.VISION_MULTIPASS_VOTES); return isNaN(n) || n < 1 ? 2 : n; })();  // v9: default 2, min 1

/** Comma-separated temperatures for multi-pass diversity */
export const MULTIPASS_TEMPERATURES = process.env.VISION_MULTIPASS_TEMPERATURES || "0,0.03";  // v9: low-temp annealing

/** L2 field crop padding ratio (default 0.4 = 40%) */
export const L2_CROP_PADDING = (() => { const n = Number(process.env.VISION_L2_PADDING); return isNaN(n) || n < 0 ? 0.4 : n; })();

/** Apply preprocessing to L2 field crops before OCR */
export const L2_PREPROCESS_ENABLED = process.env.VISION_L2_PREPROCESS !== "false";

// ---- v8: Multi-model routing (CE-based) ----

/** Fallback model when CE is high (use higher-quality model) */
export const CE_FALLBACK_MODEL = process.env.VISION_CE_FALLBACK_MODEL || "qwen-vl-max";

/** Whether to enable CE-based model routing */
export const ENABLE_CE_ROUTING = process.env.VISION_ENABLE_CE_ROUTING !== "false";

// ---- v8.2: Quality Guards ----

/** Minimum WebP quality for document/OCR prompts (default 92, range 50-100) */
export const MIN_OCR_QUALITY = (() => { const n = Number(process.env.VISION_MIN_OCR_QUALITY); return isNaN(n) || n < 50 || n > 100 ? 92 : n; })();

// ---- v8.2: Thinking + Preprocessing ----

/** Allow preprocessing when thinking is enabled (default false - skip binarization with thinking) */
export const PREPROCESS_WITH_THINKING = process.env.VISION_PREPROCESS_WITH_THINKING === "true";

// ---- v8.2: Progressive Timeout Fallback ----

/** L1 layout analysis timeout in ms (default 60000). Falls back to no-thinking on timeout. */
export const L1_TIMEOUT_MS = (() => { const n = Number(process.env.VISION_L1_TIMEOUT_MS); return isNaN(n) || n <= 0 ? 60000 : n; })();

/** Only run L4 voting on fields with CE > CE_THRESHOLD (default true). */
export const L4_ONLY_HIGH_CE = process.env.VISION_L4_ONLY_HIGH_CE !== "false";

// ---- v8.2: Second-Pass Correction ----

/** Enable two-stage extraction with second-pass correction */
export const SECOND_PASS_ENABLED = process.env.VISION_SECOND_PASS_ENABLED === "true";  // v9: disabled by default

/** Model for second-pass correction (default qwen3-vl-flash for speed/cost) */
export const SECOND_PASS_MODEL = process.env.VISION_SECOND_PASS_MODEL || "qwen3-vl-flash";

// ---- v8.3: Image-based PDF Optimization ----

/** For image-based PDFs, keep PNG lossless (no WebP/JPEG re-compression) */
export const IMAGE_PDF_LOSSLESS = process.env.VISION_IMAGE_PDF_LOSSLESS !== "false";

/** Max image dimension (width or height) in pixels. 0 = no limit. Default 4096. */
export const IMAGE_MAX_DIMENSION = (() => { const n = Number(process.env.VISION_IMAGE_MAX_DIMENSION); return isNaN(n) || n < 0 ? 4096 : n; })();

/** Max bytes per single image before compression fallback. Default 20MB. */
export const IMAGE_MAX_BYTES = (() => { const n = Number(process.env.VISION_IMAGE_MAX_BYTES); return isNaN(n) || n <= 0 ? 20 * 1024 * 1024 : n; })();

/** Compression fallback format when image exceeds limits: jpeg | webp */
export const IMAGE_COMPRESSION_FALLBACK: "jpeg" | "webp" =
  (process.env.VISION_IMAGE_COMPRESSION_FALLBACK as "jpeg" | "webp") || "jpeg";

/** JPEG quality for compression fallback (1-100, default 92) */
export const IMAGE_JPEG_QUALITY = (() => { const n = Number(process.env.VISION_IMAGE_JPEG_QUALITY); return isNaN(n) || n < 50 || n > 100 ? 92 : n; })();


// Log startup config
export function logConfig() {
  console.error(
    `[vision-mcp] v10.0.0 | ${MODEL} @ ${BASE_URL} | provider=${PROVIDER_TYPE} | ` +
    `concurrency=${CONCURRENCY} adaptive=${ADAPTIVE_CONCURRENCY} max=${CONCURRENCY_MAX} | ` +
    `webp_q=${WEBP_QUALITY} max_w=${MAX_IMAGE_WIDTH} | ` +
    `temp=${TEMPERATURE ?? "default"} top_p=${TOP_P ?? "default"} | ` +
    `budget=${TOKEN_BUDGET_ENABLED ? TOKEN_BUDGET_K + "K" : "off"} | ` +
    `thinking=${ENABLE_THINKING}${THINKING_BUDGET > 0 ? ` budget=${THINKING_BUDGET}` : ""} | ` +
    `vl_high_res=${VL_HIGH_RES_ENABLED} max_pix_text=${MAX_PIXELS_TEXT} dense=${MAX_PIXELS_DENSE} | ` +
    `preprocess=${ENABLE_PREPROCESSING} layered=${ENABLE_LAYERED_EXTRACTION} sc=${ENABLE_SELF_CONSISTENCY} | ` +
    `CE=${ENABLE_CE} threshold=${CE_THRESHOLD} votes=${CE_VOTES} | ` +
    `post_process=${ENABLE_POST_PROCESS} cross_field=${ENABLE_CROSS_FIELD} | ` +
    `edge_enhance=${ENABLE_EDGE_ENHANCE} adaptive_thresh=${ENABLE_ADAPTIVE_THRESHOLD} | ` +
    `ocr_model=${OCR_MODEL} render_cache=${RENDER_CACHE_ENABLED} | ` +
    `cost=${COST_POLICY} cache=${CACHE_POLICY} cost_breakdown=${RETURN_COST_BREAKDOWN} max_unverified_required=${MAX_UNVERIFIED_REQUIRED_FIELDS} | ` +
    `strategy=${EXTRACTION_STRATEGY} scanned=${SCANNED_STRATEGY} multipass=${MULTIPASS_ENABLED}@${MULTIPASS_VOTES} | ` +
    `l2_pad=${L2_CROP_PADDING} l2_pp=${L2_PREPROCESS_ENABLED} | ` +
    `quality: min_ocr_q=${MIN_OCR_QUALITY} prep_think=${PREPROCESS_WITH_THINKING} l1_to=${L1_TIMEOUT_MS}ms l4_ce=${L4_ONLY_HIGH_CE} sp=${SECOND_PASS_ENABLED}@${SECOND_PASS_MODEL} | ` +
    `image_pdf: lossless=${IMAGE_PDF_LOSSLESS} max_dim=${IMAGE_MAX_DIMENSION} max_bytes=${(IMAGE_MAX_BYTES/1024/1024).toFixed(0)}MB fallback=${IMAGE_COMPRESSION_FALLBACK} jpg_q=${IMAGE_JPEG_QUALITY}`
  );
}
