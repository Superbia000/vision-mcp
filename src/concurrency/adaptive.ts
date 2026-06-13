/**
 * Vision-MCP v7: Adaptive Concurrency Controller
 *
 * Dynamically adjusts parallel API call count based on:
 * - Average latency (moving average)
 * - Error rate (especially 429 rate limits)
 * - Current success rate
 *
 * Rules:
 * - Start at initial concurrency (default 20)
 * - Success + low latency -> increment (up to max, default 50)
 * - 429 errors -> exponential backoff
 * - 5xx errors -> moderate reduction
 * - Reevaluate every few seconds
 */

export interface ConcurrencyState {
  current: number;
  min: number;
  max: number;
  initial: number;
  latencyAvg: number;
  errorCount: number;
  successCount: number;
  lastAdjustment: number;
}

export class AdaptiveConcurrency {
  private state: ConcurrencyState;
  private enabled: boolean;

  constructor(
    initial: number = 20,
    max: number = 50,
    min: number = 2,
    enabled: boolean = true
  ) {
    this.state = {
      current: initial,
      min,
      max,
      initial,
      latencyAvg: 0,
      errorCount: 0,
      successCount: 0,
      lastAdjustment: Date.now(),
    };
    this.enabled = enabled;
  }

  /** Get current concurrency limit */
  get current(): number {
    return this.enabled ? this.state.current : this.state.initial;
  }

  /** Report a successful API call with its latency */
  reportSuccess(latencyMs: number): void {
    if (!this.enabled) return;
    const alpha = 0.3; // EMA smoothing factor
    this.state.latencyAvg =
      this.state.latencyAvg === 0
        ? latencyMs
        : this.state.latencyAvg * (1 - alpha) + latencyMs * alpha;
    this.state.successCount++;
    this.maybeAdjust();
  }

  /** Report a failed API call */
  reportError(status: number): void {
    if (!this.enabled) return;
    this.state.errorCount++;

    if (status === 429) {
      // Rate limit: aggressive backoff
      this.state.current = Math.max(
        this.state.min,
        Math.floor(this.state.current / 2)
      );
      console.error(
        `[concurrency] Rate limited (429): reduced to ${this.state.current}`
      );
    } else if (status >= 500) {
      // Server error: moderate reduction
      this.state.current = Math.max(
        this.state.min,
        Math.floor(this.state.current * 0.7)
      );
      console.error(
        `[concurrency] Server error (${status}): reduced to ${this.state.current}`
      );
    }
    this.state.lastAdjustment = Date.now();
  }

  /** Check if we should increase concurrency */
  private maybeAdjust(): void {
    const now = Date.now();
    // Only adjust every 5 seconds
    if (now - this.state.lastAdjustment < 5000) return;

    const total = this.state.successCount + this.state.errorCount;
    if (total < 10) return; // Not enough data

    const errorRate = this.state.errorCount / total;

    if (errorRate < 0.05 && this.state.latencyAvg < 3000) {
      // Good conditions: increase
      this.state.current = Math.min(
        this.state.max,
        Math.floor(this.state.current * 1.3)
      );
      console.error(
        `[concurrency] Good conditions (err=${(errorRate * 100).toFixed(1)}%, lat=${this.state.latencyAvg.toFixed(0)}ms): increased to ${this.state.current}`
      );
    } else if (errorRate > 0.15 || this.state.latencyAvg > 10000) {
      // Bad conditions: decrease
      this.state.current = Math.max(
        this.state.min,
        Math.floor(this.state.current * 0.7)
      );
      console.error(
        `[concurrency] Degraded (err=${(errorRate * 100).toFixed(1)}%, lat=${this.state.latencyAvg.toFixed(0)}ms): decreased to ${this.state.current}`
      );
    }

    // Reset counters
    this.state.errorCount = 0;
    this.state.successCount = 0;
    this.state.lastAdjustment = now;
  }

  /** Get current state for debugging */
  getState(): ConcurrencyState {
    return { ...this.state };
  }

  /** Manually set concurrency (override) */
  setConcurrency(n: number): void {
    this.state.current = Math.max(this.state.min, Math.min(this.state.max, n));
  }

  /** Enable/disable adaptive mode */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.state.current = this.state.initial;
    }
  }
}

/**
 * Create a pLimit-compatible function that uses adaptive concurrency.
 */
export function createAdaptiveLimit(
  adaptive: AdaptiveConcurrency
): <T>(fn: () => Promise<T>) => Promise<T> {
  // Dynamic import p-limit since it's ESM
  let pLimitFn: any;
  let currentLimit: any;
  let currentConcurrency = 0;

  async function getLimit(): Promise<any> {
    const concurrency = adaptive.current;
    if (concurrency !== currentConcurrency) {
      const pLimitModule = await import("p-limit");
      pLimitFn = pLimitModule.default;
      currentLimit = pLimitFn(concurrency);
      currentConcurrency = concurrency;
    }
    return currentLimit;
  }

  return async <T>(fn: () => Promise<T>): Promise<T> => {
    const limit = await getLimit();
    return limit(fn);
  };
}
