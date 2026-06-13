/**
 * Vision-MCP v7: General helper utilities
 */

import { SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS } from "../config/constants.js";

export function parsePageRange(raw: string, totalPages: number): number[] {
  const pages: number[] = [];
  for (const part of raw.split(",")) {
    const t = part.trim();
    if (t.includes("-")) {
      const [s, e] = t.split("-");
      for (let i = Math.max(1, +s); i <= Math.min(totalPages, +e); i++)
        pages.push(i);
    } else {
      const n = +t;
      if (n >= 1 && n <= totalPages) pages.push(n);
    }
  }
  return [...new Set(pages)].sort((a, b) => a - b);
}

export function extToMime(ext: string): string {
  return (
    {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".webp": "image/webp",
      ".gif": "image/gif",
      ".mp4": "video/mp4",
      ".mpeg": "video/mpeg",
      ".mov": "video/quicktime",
      ".avi": "video/x-msvideo",
      ".flv": "video/x-flv",
      ".mpg": "video/mpeg",
      ".webm": "video/webm",
      ".wmv": "video/x-ms-wmv",
      ".3gpp": "video/3gpp",
    } as Record<string, string>
  )[ext] || "application/octet-stream";
}

export function chunkArray<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

export function isDocExtractionPrompt(prompt: string): boolean {
  return /提取|OCR|識別|掃描|parse|extract|document|表格|table|發票|invoice|收據|receipt|pdf|markdown|html|qwenvl/i.test(prompt);
}

export function isHandwritingPrompt(prompt: string): boolean {
  return /手寫|手写|handwrit|手書き|cursive|manuscript|手稿|筆跡|笔迹/i.test(prompt);
}

export function isFieldExtractionPrompt(prompt: string): boolean {
  return /欄位|字段|field|extract.*(value|data|information|detail)|key.*value|structured|json|table.*extract|提取.*欄|擷取.*欄/i.test(prompt);
}

/** Detect if the file extension is a supported image type */
export function isImageExt(ext: string): boolean {
  return SUPPORTED_IMAGE_EXTS.has(ext.toLowerCase());
}

/** Detect if the file extension is a supported video type */
export function isVideoExt(ext: string): boolean {
  return SUPPORTED_VIDEO_EXTS.has(ext.toLowerCase());
}

/** Simple LRU cache for rendered pages */
export class LRUCache<V> {
  private map = new Map<string, V>();
  private max: number;

  constructor(max: number) {
    this.max = max;
  }

  get(key: string): V | undefined {
    const val = this.map.get(key);
    if (val !== undefined) {
      this.map.delete(key);
      this.map.set(key, val);
    }
    return val;
  }

  set(key: string, val: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, val);
    if (this.map.size > this.max) {
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first);
    }
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  clear(): void {
    this.map.clear();
  }
}

export function json(d: any) {
  return { content: [{ type: "text" as const, text: JSON.stringify(d, null, 2) }] };
}

export function error(m: string) {
  return { content: [{ type: "text" as const, text: `Error: ${m}` }] };
}
