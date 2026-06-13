/**
 * Vision-MCP v7: Exponential backoff retry helper
 */

import { RETRY_MAX, RETRY_BASE_MS } from "../config/constants.js";

export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  label: string
): Promise<T> {
  let lastErr: any;
  for (let i = 0; i <= RETRY_MAX; i++) {
    try {
      return await fn();
    } catch (err: any) {
      lastErr = err;
      const status = err?.status || err?.statusCode || 0;
      if (i < RETRY_MAX && (status === 429 || status >= 500)) {
        const delay = RETRY_BASE_MS * Math.pow(2, i) + Math.random() * 500;
        console.error(
          `[retry] ${label} retry ${i + 1}/${RETRY_MAX} after ${Math.round(delay)}ms (HTTP ${status})`
        );
        await new Promise((r) => setTimeout(r, delay));
      } else {
        break;
      }
    }
  }
  throw lastErr;
}

/** Determine if an error is retryable (rate limit or server error) */
export function isRetryable(err: any): boolean {
  const status = err?.status || err?.statusCode || 0;
  return status === 429 || status >= 500;
}
