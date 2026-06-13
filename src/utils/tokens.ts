/**
 * Vision-MCP v7: Token estimation helpers
 * Based on Alibaba Cloud official formulas
 */

export function estimateImageTokens(width: number, height: number): number {
  const factor = 32;
  const hBar = Math.round(height / factor) * factor;
  const wBar = Math.round(width / factor) * factor;
  const minPixels = 4 * factor * factor;
  let maxPixels = 16384 * factor * factor;
  let h = hBar, w = wBar;
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.floor(height / beta / factor) * factor;
    w = Math.floor(width / beta / factor) * factor;
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (h * w));
    h = Math.ceil(h * beta / factor) * factor;
    w = Math.ceil(w * beta / factor) * factor;
  }
  return Math.round((h * w) / (factor * factor)) + 2;
}

export function estimateVideoTokens(
  nFrames: number,
  frameWidth: number,
  frameHeight: number
): number {
  const perFrame = estimateImageTokens(frameWidth, frameHeight) - 2;
  return Math.ceil(nFrames / 2) * perFrame + 2;
}

export function videoFpsToFrameEstimate(
  fileSizeBytes: number,
  fps: number
): number {
  const estBitrate = (5 * 1024 * 1024) / 8;
  const estDurationSec = fileSizeBytes / estBitrate;
  return Math.round(estDurationSec * fps);
}

/**
 * Pack rendered images into groups that stay within per-call token budget.
 */
export function packImagesByTokenBudget(
  images: { page: number; buffer: Buffer; mime: string; width: number; height: number }[],
  budgetK: number
): { page: number; buffer: Buffer; mime: string; width: number; height: number }[][] {
  const budgetTokens = budgetK * 1000;
  const packs: typeof images[] = [];
  let currentPack: typeof images = [];
  let currentTokens = 0;

  for (const img of images) {
    const tokens = estimateImageTokens(img.width, img.height);
    if (currentTokens + tokens > budgetTokens && currentPack.length > 0) {
      packs.push(currentPack);
      currentPack = [];
      currentTokens = 0;
    }
    currentPack.push(img);
    currentTokens += tokens;
  }
  if (currentPack.length > 0) packs.push(currentPack);
  return packs;
}
