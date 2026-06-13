/**
 * Vision-MCP v7: Video analysis tools (analyze_video + analyze_video_chunked)
 */

import { statSync, readFileSync, unlinkSync, rmdirSync } from "fs";
import { extname } from "path";
import type { VisionProvider } from "../providers/base.js";
import {
  MAX_VIDEO_BYTES_LOCAL, VIDEO_FPS_DEFAULT, VIDEO_MAX_FRAMES,
  VIDEO_CHUNK_DURATION_SEC,
  MODEL } from "../config/constants.js";
import { isVideoExt, extToMime } from "../utils/helpers.js";
import {
  getVideoDurationSec, splitVideoIntoChunks, cleanupChunks,
} from "../rendering/video.js";
import { enhancePrompt } from "./helpers.js";

export async function handleAnalyzeVideo(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const path = args.video_path as string;
  const prompt = args.prompt as string;
  const mt = (args.max_tokens as number) ?? 4096;
  const enableThinking = args.enable_thinking as boolean | undefined;
  const thinkingBudget = args.thinking_budget as number | undefined;
  const fps = (args.fps as number) || VIDEO_FPS_DEFAULT;
  const nframes = args.nframes as number | undefined;
  const temperature = args.temperature as number | undefined;
  const topP = args.top_p as number | undefined;

  if (!path) return JSON.stringify({ error: "video_path required" });
  if (!prompt) return JSON.stringify({ error: "prompt required" });

  const ext = extname(path).toLowerCase();
  if (!isVideoExt(ext)) return JSON.stringify({ error: `Unsupported format: ${ext}` });

  const info = statSync(path);
  const maxVidBytes = provider.type === "qwen" ? MAX_VIDEO_BYTES_LOCAL : 80 * 1024 * 1024;
  if (info.size > maxVidBytes) return JSON.stringify({
    error: `Too large: ${(info.size / 1024 / 1024).toFixed(1)} MB. Use analyze_video_chunked.`
  });

  try {
    const enhancedPrompt = enhancePrompt(prompt);
    const r = await provider.chatVideo(path, enhancedPrompt, {
      model: MODEL,
      max_tokens: mt,
      enable_thinking: enableThinking,
      thinking_budget: thinkingBudget,
      fps,
      nframes,
      temperature,
      top_p: topP,
    });

    return JSON.stringify({
      success: true,
      text: r.text,
      reasoning: r.reasoning,
      input_tokens: r.it,
      output_tokens: r.ot,
    });
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}

export async function handleAnalyzeVideoChunked(
  provider: VisionProvider,
  args: any
): Promise<string> {
  const videoPath = args.video_path as string;
  const prompt = args.prompt as string;
  const mt = (args.max_tokens as number) ?? 4096;
  const enableThinking = args.enable_thinking as boolean | undefined;
  const thinkingBudget = args.thinking_budget as number | undefined;
  const fps = (args.fps as number) || VIDEO_FPS_DEFAULT;
  const nframes = args.nframes as number | undefined;
  const temperature = args.temperature as number | undefined;
  const topP = args.top_p as number | undefined;

  if (!videoPath) return JSON.stringify({ error: "video_path required" });
  if (!prompt) return JSON.stringify({ error: "prompt required" });

  const ext = extname(videoPath).toLowerCase();
  if (!isVideoExt(ext)) return JSON.stringify({ error: `Unsupported format: ${ext}` });

  try {
    const info = statSync(videoPath);
    const fileSizeMB = (info.size / 1024 / 1024).toFixed(1);
    console.error(`[video_chunked] ${fileSizeMB}MB, fps=${fps}`);

    const maxBytes = provider.type === "qwen" ? MAX_VIDEO_BYTES_LOCAL : 80 * 1024 * 1024;

    if (info.size <= maxBytes) {
      console.error(`[video_chunked] Within size limit, single file processing...`);
      const r = await provider.chatVideo(videoPath, prompt, {
        model: MODEL,
        max_tokens: mt,
        enable_thinking: enableThinking,
        thinking_budget: thinkingBudget,
        fps,
        nframes,
        temperature,
        top_p: topP,
      });
      return JSON.stringify({ mode: "single", success: true, text: r.text, reasoning: r.reasoning, input_tokens: r.it, output_tokens: r.ot });
    }

    const chunkDurationSec = (args.chunk_duration_sec as number) || VIDEO_CHUNK_DURATION_SEC;
    const shouldAggregate = args.aggregate !== false;
    console.error(`[video_chunked] Splitting into ${chunkDurationSec}s chunks...`);

    const split = await splitVideoIntoChunks(videoPath, chunkDurationSec);
    const chunkResults: any[] = [];
    let totalInput = 0, totalOutput = 0;

    try {
      for (let i = 0; i < split.chunks.length; i++) {
        const chunkPath = split.chunks[i];
        console.error(`[video_chunked] Chunk ${i + 1}/${split.chunks.length}`);
        try {
          const r = await provider.chatVideo(chunkPath, `[Video chunk ${i + 1}/${split.chunks.length}]\n${prompt}`, {
            model: MODEL,
            max_tokens: mt,
            enable_thinking: enableThinking,
            thinking_budget: thinkingBudget,
            fps,
            nframes,
            temperature,
            top_p: topP,
          });
          totalInput += r.it || 0;
          totalOutput += r.ot || 0;
          chunkResults.push({ chunk: i + 1, success: true, text: r.text, reasoning: r.reasoning, input_tokens: r.it, output_tokens: r.ot });
        } catch (err: any) {
          chunkResults.push({ chunk: i + 1, success: false, text: "", error: err.message });
        }
      }
    } finally {
      cleanupChunks(split.chunks, split.dir);
    }

    const ok = chunkResults.filter((r) => r.success).length;
    return JSON.stringify({
      mode: "chunked",
      success: ok > 0,
      chunks: split.chunks.length,
      successful: ok,
      failed: split.chunks.length - ok,
      aggregate_text: shouldAggregate ? chunkResults.filter((r) => r.success).map((r) => `[Chunk ${r.chunk}]\n${r.text}`).join("\n\n") : undefined,
      results: chunkResults,
      input_tokens: totalInput || undefined,
      output_tokens: totalOutput || undefined,
    });
  } catch (err: any) {
    return JSON.stringify({ error: err.message });
  }
}
