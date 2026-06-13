/**
 * Vision-MCP v7: Tools for PDF info, batch status, tokens, video, and extraction
 */

import { statSync, readFileSync } from "fs";
import { extname } from "path";
import type { VisionProvider } from "../providers/base.js";
import { getPdfInfo } from "../rendering/pdf.js";
import { getBatchResults } from "../batch/manager.js";
import { isImageExt, isVideoExt, extToMime } from "../utils/helpers.js";

// ---- get_pdf_info ----
export async function handlePdfInfo(args: any): Promise<string> {
  const path = args.pdf_path as string;
  if (!path) return JSON.stringify({ error: "pdf_path required" });
  try {
    const fsize = statSync(path).size;
    const info = await getPdfInfo(path, fsize);
    return JSON.stringify(info, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- get_batch_status ----
export async function handleBatchStatus(provider: VisionProvider, args: any): Promise<string> {
  const batchId = args.batch_id as string;
  if (!batchId) return JSON.stringify({ error: "batch_id required" });
  try {
    const result = await getBatchResults(provider, batchId);
    return JSON.stringify(result, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- get_batch_status_all ----
export async function handleBatchStatusAll(provider: VisionProvider, args: any): Promise<string> {
  const batchIds = args.batch_ids as string[];
  if (!batchIds || !batchIds.length) return JSON.stringify({ error: "batch_ids required" });
  try {
    const results = await Promise.all(
      batchIds.map(async (bid: string) => {
        try {
          const batch = await provider.getBatch(bid);
          return {
            batch_id: bid,
            status: batch.status,
            requested: batch.request_counts?.total || 0,
            completed: batch.request_counts?.completed || 0,
            failed: batch.request_counts?.failed || 0,
            pending: batch.status === "queued" || batch.status === "validating" || batch.status === "in_progress",
            output_file_id: batch.output_file_id,
            errors: batch.errors || null,
          };
        } catch (err: any) {
          return { batch_id: bid, status: "error", error: err.message };
        }
      })
    );
    const summary = {
      total_batches: results.length,
      completed: results.filter((r) => r.status === "completed").length,
      in_progress: results.filter((r) => r.status === "queued" || r.status === "in_progress" || r.status === "validating").length,
      failed: results.filter((r) => r.status === "failed" || r.status === "error" || r.status === "expired" || r.status === "cancelled").length,
      total_requested: results.reduce((s, r) => s + (r.requested || 0), 0),
      total_completed_req: results.reduce((s, r) => s + (r.completed || 0), 0),
    };
    return JSON.stringify({ summary, batches: results }, null, 2);
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

// ---- estimate_tokens ----
export async function handleEstimateTokens(args: any): Promise<string> {
  const filePath = args.file_path as string;
  const pagesRaw = (args.pages as string) || "1";
  const imgMaxWidth = (args.max_image_width as number) ?? 2048;
  const fps = (args.fps as number) || 1;

  if (!filePath) return JSON.stringify({ error: "file_path required" });

  try {
    const ext = extname(filePath).toLowerCase();
    const info = statSync(filePath);

    if (isImageExt(ext)) {
      const sharp = await import("sharp");
      const meta = await sharp.default(readFileSync(filePath)).metadata();
      const { estimateImageTokens } = await import("../utils/tokens.js");
      const w = meta.width || 0, h = meta.height || 0;
      const targetW = Math.min(w, imgMaxWidth);
      const scale = targetW / w;
      const tokens = estimateImageTokens(Math.round(w * scale), Math.round(h * scale));
      return JSON.stringify({ type: "image", file_mb: +(info.size / 1024 / 1024).toFixed(2), dimensions: `${w}x${h}`, estimated_tokens: tokens });
    }

    if (ext === ".pdf") {
      const { getPdfPageCount, renderPageSmart } = await import("../rendering/pdf.js");
      const { parsePageRangeDetailed } = await import("../utils/helpers.js");
      const { estimateImageTokens } = await import("../utils/tokens.js");
      const total = await getPdfPageCount(filePath);
      const parsed = parsePageRangeDetailed(pagesRaw, total);
      if (parsed.error) return JSON.stringify({ error: parsed.error });
      const nums = parsed.pages;
      let totalTokens = 0;
      const perPage: any[] = [];
      const sampleSize = Math.min(nums.length, 10);
      for (let i = 0; i < sampleSize; i++) {
        const img = await renderPageSmart(filePath, nums[i], imgMaxWidth, false, false, false, {
          renderScale: args.render_scale,
          maxPixels: args.max_pixels,
          losslessMode: args.lossless_mode !== false,
        });
        const tokens = estimateImageTokens(img.width, img.height);
        perPage.push({ page: nums[i], dimensions: `${img.width}x${img.height}`, estimated_tokens: tokens });
        totalTokens += tokens;
      }
      const avgTokens = perPage.length ? Math.round(totalTokens / perPage.length) : 0;
      return JSON.stringify({
        type: "pdf", file_mb: +(info.size / 1024 / 1024).toFixed(2), total_pages: total,
        requested_pages: nums.length, sampled_pages: sampleSize,
        avg_tokens_per_page: avgTokens, estimated_total_tokens: avgTokens * nums.length,
        per_page_sample: perPage,
      });
    }

    if (isVideoExt(ext)) {
      const { estimateVideoTokens, videoFpsToFrameEstimate } = await import("../utils/tokens.js");
      const { VIDEO_MAX_FRAMES } = await import("../config/constants.js");
      const estimatedFrames = Math.min(Math.round(videoFpsToFrameEstimate(info.size, fps)), 2000);
      const estTokens = estimateVideoTokens(estimatedFrames, imgMaxWidth, Math.round(imgMaxWidth * 0.5625));
      return JSON.stringify({
        type: "video", file_mb: +(info.size / 1024 / 1024).toFixed(2), fps,
        estimated_frames: estimatedFrames, estimated_tokens: estTokens,
        note: "Rough estimate. Actual tokens depend on video resolution and model frame extraction logic.",
      });
    }

    return JSON.stringify({ error: `Unsupported file type: ${ext}` });
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}
