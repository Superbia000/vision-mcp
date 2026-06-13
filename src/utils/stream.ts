/**
 * Vision-MCP v7: Streaming utilities
 */

import { PageResult } from "../config/types.js";

/** Callback type for streaming page results */
export type PageResultCallback = (result: PageResult) => void;

/** Batch a list of pages and stream results incrementally */
export async function streamPages<T>(
  items: T[],
  processor: (item: T) => Promise<PageResult>,
  onResult: PageResultCallback,
  concurrency: number
): Promise<PageResult[]> {
  const results: PageResult[] = [];
  const queue = [...items];
  let active = 0;

  return new Promise<PageResult[]>((resolve, reject) => {
    function next() {
      while (active < concurrency && queue.length > 0) {
        const item = queue.shift()!;
        active++;
        processor(item)
          .then((result) => {
            results.push(result);
            onResult(result);
            active--;
            next();
          })
          .catch((err) => {
            const errResult: PageResult = {
              page: 0,
              success: false,
              text: "",
              error: err.message,
            };
            results.push(errResult);
            onResult(errResult);
            active--;
            next();
          });
      }
      if (active === 0 && queue.length === 0) {
        resolve(results);
      }
    }
    next();
  });
}
