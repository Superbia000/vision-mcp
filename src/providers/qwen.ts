/**
 * Vision-MCP v7: Qwen / DashScope Vision Provider
 */

import OpenAI from "openai";
import { createReadStream, readFileSync } from "fs";
import { VisionProvider, type ChatParams, type VideoChatParams } from "./base.js";
import type { VisionResponse } from "../config/types.js";
import { ENABLE_THINKING, THINKING_BUDGET, TEMPERATURE, TOP_P } from "../config/constants.js";
import { extToMime } from "../utils/helpers.js";
import { extname } from "path";

export class QwenProvider extends VisionProvider {
  readonly type = "qwen" as const;
  private client: OpenAI;

  constructor(baseUrl: string, apiKey: string) {
    super();
    this.client = new OpenAI({ apiKey, baseURL: baseUrl, timeout: 300_000, maxRetries: 2 });
  }

  getClient(): OpenAI {
    return this.client;
  }

  protected buildChatParams(overrides: Partial<ChatParams>): any {
    const base = super.buildChatParams(overrides);

    const doThink = overrides.enable_thinking !== undefined ? overrides.enable_thinking : ENABLE_THINKING;
    const budget = overrides.thinking_budget !== undefined ? overrides.thinking_budget : THINKING_BUDGET;

    base.enable_thinking = doThink;
    if (doThink && budget > 0) base.thinking_budget = budget;

    if (overrides.vl_high_resolution_images !== undefined) {
      base.vl_high_resolution_images = overrides.vl_high_resolution_images;
    }
    if (overrides.max_pixels !== undefined) base.max_pixels = overrides.max_pixels;
    if (overrides.min_pixels !== undefined) base.min_pixels = overrides.min_pixels;

    // Temperature / top_p
    if (overrides.temperature === undefined && TEMPERATURE !== undefined) base.temperature = TEMPERATURE;
    if (overrides.top_p === undefined && TOP_P !== undefined) base.top_p = TOP_P;

    return base;
  }

  async chat(params: ChatParams): Promise<VisionResponse> {
    const r = await this.client.chat.completions.create(this.buildChatParams(params) as any);
    return this.extractResponse(r);
  }

  async chatMultiImage(
    images: { buf: Buffer; mime: string }[],
    text: string,
    params: Partial<ChatParams>
  ): Promise<VisionResponse> {
    const content: any[] = images.map((img) => ({
      type: "image_url",
      image_url: { url: `data:${img.mime};base64,${img.buf.toString("base64")}` },
    }));
    content.push({ type: "text", text });

    const fullParams: ChatParams = {
      model: params.model!,
      max_tokens: params.max_tokens!,
      messages: [{ role: "user", content }],
      ...params,
    };

    return this.chat(fullParams);
  }

  async chatVideo(
    videoPath: string,
    text: string,
    params: Partial<VideoChatParams>
  ): Promise<VisionResponse> {
    const mime = extToMime(extname(videoPath).toLowerCase());
    const dataUrl = `data:${mime};base64,${readFileSync(videoPath).toString("base64")}`;
    const videoContent: any = { type: "video_url", video_url: { url: dataUrl } };

    if (params.fps !== undefined) (videoContent.video_url as any).fps = params.fps;
    if (params.nframes !== undefined) (videoContent.video_url as any).nframes = params.nframes;

    const fullParams: ChatParams = {
      model: params.model!,
      max_tokens: params.max_tokens!,
      messages: [{ role: "user", content: [videoContent, { type: "text", text }] }],
      ...params,
    };

    return this.chat(fullParams);
  }

  async createBatch(jsonlContent: string): Promise<string> {
    // Qwen/DashScope batch API
    const { join } = await import("path");
    const { writeFileSync, unlinkSync } = await import("fs");
    const { tmpdir } = await import("os");

    const jsonlFile = join(tmpdir(), `vision-batch-${Date.now()}.jsonl`);
    writeFileSync(jsonlFile, jsonlContent, "utf-8");
    const jsonlUpload = await this.client.files.create({
      file: createReadStream(jsonlFile),
      purpose: "batch" as any,
    });
    const batch = await (this.client as any).batches.create({
      input_file_id: jsonlUpload.id,
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
    });
    unlinkSync(jsonlFile);
    return batch.id;
  }

  async getBatch(batchId: string): Promise<any> {
    return (this.client as any).batches.retrieve(batchId);
  }

  async getBatchResults(outputFileId: string): Promise<string> {
    const fileContent = await this.client.files.content(outputFileId);
    return fileContent.text();
  }
}
