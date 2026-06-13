/**
 * Vision-MCP v7: Batch Job Manager
 */

import { writeFileSync, createReadStream, unlinkSync, mkdirSync, rmdirSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import type { VisionProvider } from "../providers/base.js";
import { BATCH_MAX_PAGES, MODEL } from "../config/constants.js";
import { chunkArray } from "../utils/helpers.js";
import { renderPageSmart } from "../rendering/pdf.js";
import { optimizeForVision } from "../rendering/image.js";
import type { PageResult, BatchResult } from "../config/types.js";

export interface BatchRenderOptions {
  renderScale?: number;
  maxPixels?: number;
  minPixels?: number;
  losslessMode?: boolean;
}

function imageContent(url: string, renderOptions: BatchRenderOptions): any {
  const item: any = { type: "image_url", image_url: { url } };
  if (renderOptions.minPixels !== undefined) item.min_pixels = renderOptions.minPixels;
  if (renderOptions.maxPixels !== undefined) item.max_pixels = renderOptions.maxPixels;
  return item;
}

function imageExtension(mime: string): string {
  if (mime === "image/png") return "png";
  if (mime === "image/jpeg") return "jpg";
  if (mime === "image/webp") return "webp";
  return "bin";
}

/** Submit a single batch job for a range of pages */
export async function submitBatchJob(
  provider: VisionProvider,
  pdfPath: string,
  pages: number[],
  prompt: string,
  maxTokens: number,
  imgQuality: number,
  maxImageWidth: number,
  enableThinking?: boolean,
  thinkingBudget?: number,
  renderOptions: BatchRenderOptions = {}
): Promise<string> {
  const tmpDir = join(
    tmpdir(),
    `vision-batch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );
  mkdirSync(tmpDir, { recursive: true });

  const jsonlLines: string[] = [];
  const batchBody: any = { model: MODEL, max_tokens: maxTokens };

  if (provider.type === "qwen") {
    // Qwen/DashScope: disable thinking for batch by default
    batchBody.enable_thinking = enableThinking === true;
    if (enableThinking && thinkingBudget && thinkingBudget > 0) {
      batchBody.thinking_budget = thinkingBudget;
    }
  }

  for (const pn of pages) {
    try {
      const img = await renderPageSmart(pdfPath, pn, maxImageWidth, false, false, false, {
        renderScale: renderOptions.renderScale,
        maxPixels: renderOptions.maxPixels,
        losslessMode: renderOptions.losslessMode !== false,
      });
      const opt = await optimizeForVision(img, {
        quality: imgQuality,
        maxWidth: maxImageWidth,
        preferLossless: renderOptions.losslessMode !== false,
      });

      // Qwen uses base64 inline, others use ms://
      if (provider.type === "qwen") {
        const imageItem = imageContent(
          `data:${opt.mime};base64,${opt.buffer.toString("base64")}`,
          renderOptions
        );
        jsonlLines.push(
          JSON.stringify({
            custom_id: `page_${pn}`,
            method: "POST",
            url: "/v1/chat/completions",
            body: {
              ...batchBody,
              messages: [
                {
                  role: "user",
                  content: [
                    imageItem,
                    { type: "text", text: `[Page ${pn}]\n${prompt}` },
                  ],
                },
              ],
            },
          })
        );
      } else {
        const tmpFile = join(tmpDir, `page_${pn}.${imageExtension(opt.mime)}`);
        writeFileSync(tmpFile, opt.buffer);
        const uploaded = await (provider as any).getClient().files.create({
          file: createReadStream(tmpFile),
          purpose: "vision" as any,
        });
        unlinkSync(tmpFile);
        jsonlLines.push(
          JSON.stringify({
            custom_id: `page_${pn}`,
            method: "POST",
            url: "/v1/chat/completions",
            body: {
              ...batchBody,
              messages: [
                {
                  role: "user",
                  content: [
                    { type: "image_url", image_url: { url: `ms://${uploaded.id}` } },
                    { type: "text", text: `[Page ${pn}]\n${prompt}` },
                  ],
                },
              ],
            },
          })
        );
      }
    } catch (err: any) {
      jsonlLines.push(
        JSON.stringify({
          custom_id: `page_${pn}`,
          method: "POST",
          url: "/v1/chat/completions",
          body: {
            model: MODEL,
            max_tokens: 1,
            messages: [{ role: "user", content: `ERROR: ${err.message}` }],
          },
        })
      );
    }
  }

  // Cleanup tmp dir
  const { readdir } = await import("fs/promises");
  try {
    for (const f of await readdir(tmpDir)) {
      try { unlinkSync(join(tmpDir, f)); } catch {}
    }
    rmdirSync(tmpDir);
  } catch {}

  const jsonlContent = jsonlLines.join("\n") + "\n";
  const batchId = await provider.createBatch(jsonlContent);
  return batchId;
}

/** Submit multiple batch jobs, auto-chunking pages */
export async function submitBatchedJobs(
  provider: VisionProvider,
  pdfPath: string,
  pages: number[],
  prompt: string,
  maxTokens: number,
  imgQuality: number,
  maxImageWidth: number,
  enableThinking?: boolean,
  thinkingBudget?: number,
  renderOptions: BatchRenderOptions = {}
): Promise<{ batchIds: string[]; chunks: number }> {
  const chunks = chunkArray(pages, BATCH_MAX_PAGES);
  const batchIds: string[] = [];

  for (const chunk of chunks) {
    const bid = await submitBatchJob(
      provider,
      pdfPath,
      chunk,
      prompt,
      maxTokens,
      imgQuality,
      maxImageWidth,
      enableThinking,
      thinkingBudget,
      renderOptions
    );
    batchIds.push(bid);
    console.error(`[batch] Submitted: ${bid} (${chunk.length} pages)`);
  }

  return { batchIds, chunks: chunks.length };
}

/** Poll and retrieve batch job results */
export async function getBatchResults(
  provider: VisionProvider,
  batchId: string
): Promise<BatchResult> {
  const batch = await provider.getBatch(batchId);
  const result: BatchResult = {
    batch_id: batch.id,
    status: batch.status,
    requested: batch.request_counts?.total || 0,
    completed: batch.request_counts?.completed || 0,
    failed: batch.request_counts?.failed || 0,
  };

  if (batch.errors) result.errors = batch.errors;
  if (batch.output_file_id) result.output_file_id = batch.output_file_id;

  if (batch.status === "completed" && batch.output_file_id) {
    const text = await provider.getBatchResults(batch.output_file_id);
    const lines = text.trim().split("\n").filter(Boolean);
    result.results = lines.map((line: string) => {
      try {
        const obj = JSON.parse(line);
        const msg = obj.response?.body?.choices?.[0]?.message?.content || "";
        const success = obj.response?.status_code === 200;
        const reasoning =
          obj.response?.body?.choices?.[0]?.message?.reasoning_content || null;
        const errMsg = !success
          ? obj.response?.body?.error?.message || `HTTP ${obj.response?.status_code}`
          : undefined;
        return {
          page: parseInt((obj.custom_id || "").replace("page_", "")) || 0,
          success,
          text: msg,
          ...(reasoning ? { reasoning } : {}),
          ...(errMsg ? { error: errMsg } : {}),
        } as PageResult;
      } catch {
        return { page: 0, success: false, text: "", error: "parse error" } as PageResult;
      }
    });
  } else {
    result.pending = batch.status === "in_progress" || batch.status === "validating" || batch.status === "queued";
  }

  return result;
}
