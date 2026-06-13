import sharp from "sharp";
import type { CostBreakdownEntry, VisionResponse } from "../config/types.js";

export async function estimateImageTokens(imageBuffer: Buffer): Promise<number | undefined> {
  try {
    const meta = await sharp(imageBuffer).metadata();
    if (!meta.width || !meta.height) return undefined;
    return Math.ceil(meta.width / 32) * Math.ceil(meta.height / 32) + 2;
  } catch {
    return undefined;
  }
}

export function recordVisionCost(
  entries: CostBreakdownEntry[],
  stage: string,
  model: string,
  response: VisionResponse | undefined,
  elapsedMs: number,
  estimatedImageTokens?: number,
  notes?: string[],
  costPolicy?: string
) {
  entries.push({
    stage,
    model,
    input_tokens: response?.it,
    output_tokens: response?.ot,
    estimated_image_tokens: estimatedImageTokens,
    elapsed_ms: elapsedMs,
    cost_policy: costPolicy,
    notes,
  });
}

export function summarizeCost(entries: CostBreakdownEntry[]) {
  return {
    total_input_tokens: sum(entries, "input_tokens"),
    total_output_tokens: sum(entries, "output_tokens"),
    total_estimated_image_tokens: sum(entries, "estimated_image_tokens"),
  };
}

function sum(entries: CostBreakdownEntry[], key: keyof CostBreakdownEntry): number | undefined {
  const total = entries.reduce((acc, entry) => {
    const value = entry[key];
    return typeof value === "number" ? acc + value : acc;
  }, 0);
  return total || undefined;
}
