export interface ToolMetrics {
  elapsed_ms: number;
  api_calls?: number;
  input_tokens?: number;
  output_tokens?: number;
  estimated_image_tokens?: number;
}

export interface ResultEnvelope {
  success: boolean;
  tool: string;
  strategy: string;
  summary?: Record<string, any>;
  results?: any;
  metrics: ToolMetrics;
  warnings?: string[];
  errors?: string[];
}

export function parseJsonText(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return { raw_text: text };
  }
}

export function content(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

export function envelope(data: Omit<ResultEnvelope, "metrics"> & { metrics?: Partial<ToolMetrics> }) {
  return {
    ...data,
    metrics: {
      elapsed_ms: data.metrics?.elapsed_ms ?? 0,
      api_calls: data.metrics?.api_calls,
      input_tokens: data.metrics?.input_tokens,
      output_tokens: data.metrics?.output_tokens,
    },
  };
}

export async function timed<T>(fn: () => Promise<T>): Promise<{ value: T; elapsedMs: number }> {
  const started = Date.now();
  const value = await fn();
  return { value, elapsedMs: Date.now() - started };
}

export function asTextEnvelope(data: ResultEnvelope) {
  return content(JSON.stringify(data, null, 2));
}

export function collectTokenMetrics(value: any): Pick<ToolMetrics, "input_tokens" | "output_tokens" | "estimated_image_tokens" | "api_calls"> {
  let input = 0;
  let output = 0;
  let estimatedImage = 0;
  let calls = 0;

  const visit = (node: any) => {
    if (!node || typeof node !== "object") return;
    if (typeof node.input_tokens === "number") input += node.input_tokens;
    if (typeof node.output_tokens === "number") output += node.output_tokens;
    if (typeof node.estimated_image_tokens === "number") estimatedImage += node.estimated_image_tokens;
    if (typeof node.input_tokens === "number" || typeof node.output_tokens === "number") calls++;
    if (node.stats && typeof node.stats.totalApiCalls === "number") calls += node.stats.totalApiCalls;
    if (Array.isArray(node)) {
      for (const item of node) visit(item);
      return;
    }
    for (const item of Object.values(node)) visit(item);
  };

  visit(value);
  return {
    input_tokens: input || undefined,
    output_tokens: output || undefined,
    estimated_image_tokens: estimatedImage || undefined,
    api_calls: calls || undefined,
  };
}
