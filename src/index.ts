/**
 * Vision-MCP v10: categorized public tool surface.
 *
 * Public MCP tools:
 * - vision_inspect
 * - vision_prepare
 * - vision_analyze
 * - vision_extract
 * - vision_jobs
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  API_KEY,
  BASE_URL,
  MODEL,
  PROVIDER_TYPE,
  logConfig,
} from "./config/constants.js";
import { QwenProvider } from "./providers/qwen.js";
import { OpenAICompatProvider } from "./providers/openai-compat.js";
import type { VisionProvider } from "./providers/base.js";
import { PUBLIC_TOOLS } from "./tools/registry.js";
import { ToolRouter } from "./tools/router-v10.js";

const provider: VisionProvider = PROVIDER_TYPE === "qwen"
  ? new QwenProvider(BASE_URL, API_KEY)
  : new OpenAICompatProvider(BASE_URL, API_KEY);

const router = new ToolRouter(provider);

const server = new Server(
  { name: "vision-mcp", version: "10.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: PUBLIC_TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const noApiTools = new Set(["vision_inspect", "vision_prepare"]);

  if (!API_KEY && !noApiTools.has(name)) {
    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          success: false,
          tool: name,
          strategy: "startup-validation",
          metrics: { elapsed_ms: 0 },
          errors: ["VISION_API_KEY not set"],
        }, null, 2),
      }],
    };
  }

  try {
    return await router.dispatch(name, args || {});
  } catch (err: any) {
    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          success: false,
          tool: name,
          strategy: "unhandled-error",
          metrics: { elapsed_ms: 0 },
          errors: [err?.message || String(err)],
        }, null, 2),
      }],
    };
  }
});

async function main() {
  if (!API_KEY) console.error("[vision-mcp] VISION_API_KEY not set.");
  console.error(`[vision-mcp] v10.0.0 | public_tools=5 | model=${MODEL} | provider=${PROVIDER_TYPE}`);
  await server.connect(new StdioServerTransport());
  logConfig();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
