/**
 * Vision-MCP v7: Qwen / DashScope Vision Provider
 */

import OpenAI from "openai";
import { createReadStream, readFileSync } from "fs";
import {
  VisionProvider,
  type ChatParams,
  type NativeOcrParams,
  type NativeOcrResult,
  type VideoChatParams,
} from "./base.js";
import type { VisionResponse } from "../config/types.js";
import { ENABLE_THINKING, THINKING_BUDGET, TEMPERATURE, TOP_P } from "../config/constants.js";
import { extToMime } from "../utils/helpers.js";
import { extname } from "path";

export class QwenProvider extends VisionProvider {
  readonly type = "qwen" as const;
  private client: OpenAI;
  private readonly baseUrl: string;
  private readonly apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    super();
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
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
    this.attachPixelControls(base, overrides);

    // Temperature / top_p
    if (overrides.temperature === undefined && TEMPERATURE !== undefined) base.temperature = TEMPERATURE;
    if (overrides.top_p === undefined && TOP_P !== undefined) base.top_p = TOP_P;

    return base;
  }

  private attachPixelControls(base: any, overrides: Partial<ChatParams>): void {
    const hasMax = overrides.max_pixels !== undefined;
    const hasMin = overrides.min_pixels !== undefined;
    if (!hasMax && !hasMin) return;

    for (const message of base.messages || []) {
      if (!Array.isArray(message.content)) continue;
      for (const item of message.content) {
        if (!item || typeof item !== "object") continue;
        if (item.type !== "image_url" && item.type !== "video_url") continue;
        if (hasMin) item.min_pixels = overrides.min_pixels;
        if (hasMax) item.max_pixels = overrides.max_pixels;
      }
    }
  }

  async chat(params: ChatParams): Promise<VisionResponse> {
    const r = await this.client.chat.completions.create(this.buildChatParams(params) as any);
    return this.extractResponse(r);
  }

  supportsNativeOcr(): boolean {
    return true;
  }

  async nativeOcr(params: NativeOcrParams): Promise<NativeOcrResult> {
    const endpoint = this.nativeOcrEndpoint();
    const imageItem: any = {
      image: `data:${params.mime};base64,${params.image.toString("base64")}`,
      enable_rotate: params.enable_rotate === true,
    };
    if (params.min_pixels !== undefined) imageItem.min_pixels = params.min_pixels;
    if (params.max_pixels !== undefined) imageItem.max_pixels = params.max_pixels;

    const content: any[] = [imageItem];
    if (params.text) content.push({ text: params.text });

    const body: any = {
      model: params.model,
      input: {
        messages: [{ role: "user", content }],
      },
      parameters: {
        ocr_options: {
          task: params.task,
        },
      },
    };

    if (params.task === "key_information_extraction" && params.resultSchema) {
      body.parameters.ocr_options.task_config = { result_schema: params.resultSchema };
    }
    if (params.max_tokens !== undefined) body.parameters.max_tokens = Math.min(Math.max(1, params.max_tokens), 8192);
    if (params.temperature !== undefined) body.parameters.temperature = params.temperature;
    if (params.top_p !== undefined) body.parameters.top_p = params.top_p;
    if (params.logprobs !== undefined) body.parameters.logprobs = params.logprobs;
    if (params.top_logprobs !== undefined) body.parameters.top_logprobs = params.top_logprobs;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const responseText = await response.text();
    if (!response.ok) {
      throw new Error(`DashScope native OCR HTTP ${response.status}: ${responseText.slice(0, 1000)}`);
    }

    let raw: any;
    try {
      raw = JSON.parse(responseText);
    } catch {
      throw new Error(`DashScope native OCR returned non-JSON response: ${responseText.slice(0, 1000)}`);
    }

    const statusCode = Number(raw.status_code ?? 200);
    if (statusCode !== 200) {
      throw new Error(`DashScope native OCR ${raw.code || statusCode}: ${raw.message || responseText.slice(0, 1000)}`);
    }

    const contentItem = raw.output?.choices?.[0]?.message?.content?.[0] || {};
    const usage = raw.usage || {};
    return {
      task: params.task,
      text: String(contentItem.text || ""),
      ocrResult: contentItem.ocr_result,
      raw,
      requestId: raw.request_id,
      it: usage.input_tokens,
      ot: usage.output_tokens,
      imageTokens: usage.image_tokens ?? usage.input_tokens_details?.image_tokens,
    };
  }

  private nativeOcrEndpoint(): string {
    const root = this.baseUrl
      .replace(/\/compatible-mode\/v1\/?$/, "")
      .replace(/\/api\/v1\/?$/, "")
      .replace(/\/$/, "");
    return `${root}/api/v1/services/aigc/multimodal-generation/generation`;
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
