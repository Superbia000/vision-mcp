import type { VisionProvider } from "../providers/base.js";
import { BASE_URL, MODEL, OCR_MODEL, PROVIDER_TYPE } from "../config/constants.js";

export interface CapabilityMatrix {
  provider: "qwen" | "openai";
  model: string;
  supportsStructuredOutput: boolean;
  supportsThinking: boolean;
  supportsBatch: boolean;
  supportsVideoUrl: boolean;
  supportsMinMaxPixels: boolean;
  supportsMultiImage: boolean;
  notes: string[];
}

export function getCapabilityMatrix(provider: VisionProvider): CapabilityMatrix {
  const isQwen = provider.type === "qwen" || PROVIDER_TYPE === "qwen";
  const isOcr = /qwen-vl-ocr/i.test(MODEL) || /qwen-vl-ocr/i.test(OCR_MODEL);
  const isChinaDashScope = BASE_URL.includes("dashscope.aliyuncs.com");
  const isIntlDashScope = BASE_URL.includes("dashscope-intl.aliyuncs.com");

  return {
    provider: provider.type,
    model: MODEL,
    supportsStructuredOutput: !isOcr,
    supportsThinking: isQwen && !isOcr,
    supportsBatch: isQwen ? isChinaDashScope : true,
    supportsVideoUrl: true,
    supportsMinMaxPixels: isQwen,
    supportsMultiImage: true,
    notes: [
      ...(isOcr ? ["Qwen-OCR uses user-message instructions only and does not support multi-turn context."] : []),
      ...(isIntlDashScope ? ["International DashScope Batch model coverage is narrower; batch auto mode may fall back to realtime."] : []),
      ...(!isChinaDashScope && isQwen ? ["Multimodal batch is only forced for known supported DashScope regions/models."] : []),
    ],
  };
}
