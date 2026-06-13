/**
 * Vision-MCP v7: Abstract Vision Provider interface
 */

import OpenAI from "openai";
import type { VisionResponse } from "../config/types.js";

export interface ChatParams {
  model: string;
  max_tokens: number;
  messages: any[];
  enable_thinking?: boolean;
  thinking_budget?: number;
  vl_high_resolution_images?: boolean;
  max_pixels?: number;
  min_pixels?: number;
  temperature?: number;
  top_p?: number;
  response_format?: { type: string };
  stream?: boolean;
}

export interface VideoChatParams extends ChatParams {
  fps?: number;
  nframes?: number;
}

export type NativeOcrTask =
  | "text_recognition"
  | "key_information_extraction"
  | "document_parsing"
  | "table_parsing"
  | "formula_recognition"
  | "multi_lan"
  | "advanced_recognition";

export interface NativeOcrParams {
  model: string;
  image: Buffer;
  mime: string;
  task: NativeOcrTask;
  text?: string;
  resultSchema?: Record<string, string>;
  max_tokens?: number;
  min_pixels?: number;
  max_pixels?: number;
  enable_rotate?: boolean;
  temperature?: number;
  top_p?: number;
  logprobs?: boolean;
  top_logprobs?: number;
}

export interface NativeOcrResult {
  task: NativeOcrTask;
  text: string;
  ocrResult?: any;
  raw: any;
  requestId?: string;
  it?: number;
  ot?: number;
  imageTokens?: number;
}

export abstract class VisionProvider {
  abstract readonly type: "qwen" | "openai";
  abstract chat(params: ChatParams): Promise<VisionResponse>;
  abstract chatMultiImage(images: { buf: Buffer; mime: string }[], text: string, params: Partial<ChatParams>): Promise<VisionResponse>;
  abstract chatVideo(videoPath: string, text: string, params: Partial<VideoChatParams>): Promise<VisionResponse>;
  abstract createBatch(jsonlContent: string): Promise<string>;
  abstract getBatch(batchId: string): Promise<any>;
  abstract getBatchResults(outputFileId: string): Promise<string>;

  supportsNativeOcr(): boolean {
    return false;
  }

  async nativeOcr(_params: NativeOcrParams): Promise<NativeOcrResult> {
    throw new Error("Provider does not support DashScope native OCR");
  }

  /** Build provider-specific chat completion params */
  protected buildChatParams(overrides: Partial<ChatParams>): any {
    const base: any = {
      model: overrides.model,
      max_tokens: overrides.max_tokens,
      messages: overrides.messages,
    };

    if (overrides.temperature !== undefined) base.temperature = overrides.temperature;
    if (overrides.top_p !== undefined) base.top_p = overrides.top_p;
    if (overrides.response_format) base.response_format = overrides.response_format;
    if (overrides.stream) base.stream = overrides.stream;

    return base;
  }

  /** Extract text + reasoning from API response */
  protected extractResponse(r: any): VisionResponse {
    const msg = r.choices?.[0]?.message;
    let text = msg?.content || "";
    const reasoning = (msg as any)?.reasoning_content || null;

    // v7: Qwen thinking models may put real answer in reasoning
    if (text.length <= 30 && reasoning && reasoning.length > 200) {
      console.error(`[provider] Content too short (${text.length}c), using reasoning (${reasoning.length}c) as text`);
      text = reasoning;
    }

    return {
      text,
      reasoning,
      it: r.usage?.prompt_tokens ?? undefined,
      ot: r.usage?.completion_tokens ?? undefined,
    };
  }
}
