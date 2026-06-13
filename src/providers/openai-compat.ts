/**
 * Vision-MCP v7: Generic OpenAI-compatible Vision Provider
 * Works with any OpenAI-compatible API (non-Qwen providers)
 */

import OpenAI from "openai";
import { createReadStream, readFileSync, unlinkSync, writeFileSync } from "fs";
import { VisionProvider, type ChatParams, type VideoChatParams } from "./base.js";
import type { VisionResponse } from "../config/types.js";
import { TEMPERATURE, TOP_P } from "../config/constants.js";
import { extToMime } from "../utils/helpers.js";
import { extname, join } from "path";
import { tmpdir } from "os";

export class OpenAICompatProvider extends VisionProvider {
  readonly type = "openai" as const;
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

    // OpenAI-compat: temperature/top_p
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
    const fileObj = await this.client.files.create({
      file: createReadStream(videoPath),
      purpose: "file-extract" as any,
    });

    try {
      const videoContent: any = {
        type: "video_url",
        video_url: { url: `ms://${fileObj.id}` },
      };
      if (params.fps !== undefined) (videoContent.video_url as any).fps = params.fps;
      if (params.nframes !== undefined) (videoContent.video_url as any).nframes = params.nframes;

      const fullParams: ChatParams = {
        model: params.model!,
        max_tokens: params.max_tokens!,
        messages: [{ role: "user", content: [videoContent, { type: "text", text }] }],
        ...params,
      };

      return this.chat(fullParams);
    } finally {
      try { await this.client.files.delete(fileObj.id); } catch { /* ignore */ }
    }
  }

  async createBatch(jsonlContent: string): Promise<string> {
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
