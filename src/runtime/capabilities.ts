import type { VisionProvider } from "../providers/base.js";
import { BASE_URL, MODEL, PROVIDER_TYPE } from "../config/constants.js";

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
  const isChinaDashScope = BASE_URL.includes("dashscope.aliyuncs.com");
  const isIntlDashScope = BASE_URL.includes("dashscope-intl.aliyuncs.com");

  return {
    provider: provider.type,
    model: MODEL,
    supportsStructuredOutput: true,
    supportsThinking: isQwen,
    supportsBatch: isQwen ? isChinaDashScope : true,
    supportsVideoUrl: true,
    supportsMinMaxPixels: isQwen,
    supportsMultiImage: true,
    notes: [
      ...(isIntlDashScope ? ["International DashScope Batch model coverage is narrower; batch auto mode may fall back to realtime."] : []),
      ...(!isChinaDashScope && isQwen ? ["Multimodal batch is only forced for known supported DashScope regions/models."] : []),
    ],
  };
}
