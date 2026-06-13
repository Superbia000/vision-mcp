/**
 * Vision-MCP v8: Shared types with CE, post-processing, and enhanced extraction
 */

import type { VisionProvider } from "../providers/base.js";

export interface PageResult {
  page: number;
  success: boolean;
  text: string;
  reasoning?: string | null;
  input_tokens?: number;
  output_tokens?: number;
  error?: string;
}

export interface ExtractedImage {
  buffer: Buffer;
  mime: string;
  width: number;
  height: number;
}

export interface OptimizedImage {
  buffer: Buffer;
  mime: string;
}

export interface VisionResponse {
  text: string;
  reasoning: string | null;
  it?: number;
  ot?: number;
}

export interface CostBreakdownEntry {
  stage: string;
  model: string;
  input_tokens?: number;
  output_tokens?: number;
  estimated_image_tokens?: number;
  elapsed_ms: number;
  cost_policy?: string;
  notes?: string[];
}

export interface PdfInfo {
  file_mb: number;
  total_pages: number;
  pages: { page: number; w_pt: number; h_pt: number }[];
}

export interface ProcessingSummary {
  strategy: string;
  pipeline: Record<string, any>;
  requested: number;
  successful: number;
  failed: number;
  elapsed_seconds: number;
  total_input_tokens?: number;
  total_output_tokens?: number;
  concurrency?: number;
}

export interface BatchResult {
  batch_id: string;
  status: string;
  requested: number;
  completed: number;
  failed: number;
  errors?: any[];
  results?: PageResult[];
}

export interface FieldSpec {
  name: string;
  labelPattern: string;
  positionHint?: string;
  formatHint?: string;
  required?: boolean;
  schema?: Record<string, any>;
  /** v8: Allowed values for validation */
  allowedValues?: string[];
  /** v8: Example value for prompt enhancement */
  example?: string;
  /** v8: Cross-field validation context (e.g. "total = subtotal + tax") */
  contextRule?: string;
}

export interface LocatedField {
  name: string;
  label: string;
  value: string;
  confidence: "high" | "medium" | "low";
  bbox?: { x: number; y: number; w: number; h: number };
}

export interface DocumentLayout {
  fields: LocatedField[];
  rawText: string;
  reasoning?: string;
}

export interface LayeredExtractionResult {
  success: boolean;
  layout: DocumentLayout;
  verifiedFields: Record<string, { value: string; confidence: string; verified: boolean }>;
  finalJson: Record<string, any>;
  errors?: string[];
  /** v8: Consensus entropy per field (lower = more reliable) */
  consensusEntropy?: Record<string, number>;
  /** v8: Post-processing corrections applied */
  postProcessCorrections?: Record<string, { original: string; corrected: string; reason: string }>;
  /** v8: Cross-field validation notes */
  crossFieldNotes?: string[];
  secondPassCorrections?: string[];
  costBreakdown?: CostBreakdownEntry[];
  routingTrace?: Record<string, string[]>;
  stats: {
    totalApiCalls: number;
    totalTokens: number;
    elapsedMs: number;
    page?: number;
    estimatedImageTokens?: number;
  };
}

export interface LosslessTextItem {
  text: string;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  confidence?: "high" | "medium" | "low";
  source?: string;
}

export interface LosslessFieldCandidate {
  name?: string;
  label?: string;
  value: string;
  bbox?: { x1: number; y1: number; x2: number; y2: number };
  confidence?: "high" | "medium" | "low";
  source?: string;
  needs_review?: boolean;
}

export interface LosslessPage {
  page: number;
  raw_markdown: string;
  raw_html?: string;
  text_items: LosslessTextItem[];
  tables: any[];
  field_candidates: LosslessFieldCandidate[];
  mapped_fields: Record<string, any>;
  unmapped_fields: LosslessFieldCandidate[];
  orphan_values: any[];
  uncertain_tokens: any[];
  review_required: boolean;
}

export interface LosslessDocumentResult {
  success: boolean;
  schema: "lossless_document_v1";
  source_path?: string;
  pages: LosslessPage[];
  finalJson: Record<string, any>;
  costBreakdown?: CostBreakdownEntry[];
  review_required: boolean;
  quality_gate: { passed: boolean; reason?: string };
  stats: {
    totalApiCalls: number;
    totalTokens: number;
    elapsedMs: number;
    pageCount: number;
  };
  errors?: string[];
}

export interface LayeredExtractionConfig {
  primaryModel: string;
  ocrModel: string;
  primaryBaseUrl: string;
  ocrBaseUrl: string;
  apiKey: string;
  enableThinking?: boolean;
  /** v8: Document type for preprocessing routing */
  docType?: DocumentType;
  /** v8: Language hint for handwriting */
  languageHint?: string;
  /** v8: Enable CE scoring */
  enableCE?: boolean;
  /** v8: CE threshold for flagging */
  ceThreshold?: number;
  /** v8: Enable post-processing corrections */
  enablePostProcess?: boolean;
  /** v11: Cost policy never downgrades quality; it only controls routing and reporting */
  costPolicy?: "quality_first" | "prefer_batch" | "realtime_only";
  /** v11: Cache policy metadata for caller/provider prompt ordering decisions */
  cachePolicy?: "auto" | "off" | "explicit";
  /** v11: Require review when more than this many required fields are unverified */
  maxUnverifiedRequiredFields?: number;
  /** v11: Return routing/cost evidence */
  returnCostBreakdown?: boolean;
  /** v11: Preserve all visible text/fields even when requested fields are supplied */
  preserveAll?: boolean;
  /** v11: Structured lossless output schema identifier */
  outputSchema?: "lossless_document_v1";
  /** Internal test seam: lets focused tests exercise routing without network/API keys */
  primaryProviderOverride?: VisionProvider;
  /** Internal test seam: lets focused tests exercise OCR fallback without network/API keys */
  ocrProviderOverride?: VisionProvider;
}

export interface FieldValidationRule {
  field: string;
  pattern?: string;
  minLength?: number;
  maxLength?: number;
  allowedValues?: string[];
  /** v8: Semantic format type for auto-correction */
  format?: "date" | "number" | "email" | "phone" | "currency" | "id_card" | "custom";
}

export interface ValidationError {
  field: string;
  value: string;
  rule: string;
  message: string;
}

export interface VoteResult {
  value: string;
  frequency: number;
  totalVotes: number;
  agreement: number;
  isUnanimous: boolean;
}

/** v8: Consensus entropy result for a single field */
export interface CEFieldResult {
  fieldName: string;
  entropy: number;
  confidence: "high" | "medium" | "low";
  votes: string[];
  agreement: number;
  finalValue: string;
  needsReview: boolean;
}

export type PdfStrategy = "concurrent" | "multi-image" | "batch";

export type PreprocessMode = "light" | "aggressive" | "auto";

export type DocumentType = "scan" | "photo" | "table" | "handwriting" | "mixed";

export interface PreprocessOptions {
  autoOrient?: boolean;
  deskew?: boolean;
  enhanceContrast?: boolean;
  sharpen?: boolean;
  grayscale?: boolean;
  targetDpi?: number;
  removeBackground?: boolean;
  quality?: number;
  /** v7: Specific document type hint */
  docType?: DocumentType;
  /** v8: Enable edge enhancement (Sobel) */
  edgeEnhance?: boolean;
  /** v8: Enable adaptive threshold binarization */
  adaptiveThreshold?: boolean;
  /** v8: Enable perspective correction */
  perspectiveCorrect?: boolean;

  /** v8.1: Detect and handle negative/inverted images */
  detectNegative?: boolean;
  /** v8.1: Auto-invert detected negative images */
  autoInvert?: boolean;
}

export interface PreprocessResult {
  buffer: Buffer;
  mime: string;
  width: number;
  height: number;
  appliedSteps: string[];
  /** v8: Detected document type */
  detectedDocType?: DocumentType;
}
