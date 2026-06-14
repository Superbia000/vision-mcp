/**
 * Vision-MCP v9: Adaptive reading prompts by document category.
 *
 * Three prompt templates selected by detectDocumentType():
 *   scan       → printed text on scanned documents
 *   photo      → camera-captured document photos
 *   handwriting/mixed → handwritten or mixed print+handwriting documents
 *
 * Design principles:
 * - Teaches the model HOW to read, never WHERE to find specific fields
 * - No field names, no spatial coordinates, no document-type assumptions
 * - Anti-hallucination rules are universal across all categories
 * - Each category adds only category-specific reading guidance
 */

import type { DocumentType } from "../config/types.js";

// ── Shared anti-hallucination rules (appended to every prompt) ──

const ANTI_HALLUCINATION = `
防幻覺規則（嚴格遵守）：
- 模糊不清的字元 → 標記為 [?]，不要猜
- 完全無法辨認的值 → 留空 ""，不要編造
- 不要根據文件類型推測內容（不能因為「這看起來像發票」就假設有某個欄位）
- 不要根據上下文補全不完整的文字
- 如果一個值看起來像常見格式但不完全吻合，報告你實際看到的，不要「修正」它`;

// ── Shared reading methodology (core of every prompt) ──

const READING_METHOD = `
閱讀方法：
- 從上到下、從左到右，逐區閱讀整個頁面
- 每遇到一個區塊（標題、地址、表格、頁尾），先理解該區塊的性質，再讀取內容
- 標籤和值通常成對出現（如「日期: 27.06.22」），讀取時先識別標籤，再逐字讀取值
- 表格區域：先理解欄位標題，再逐行讀取，每格獨立判斷

逐字閱讀規則：
- 每個字元都要親眼確認，包括標點符號和空格
- 數字：逐位確認，不解讀、不進位、不換算
- 日期：保留原始格式（DD.MM.YY 就是 DD.MM.YY，不要轉換）
- 英文字母：注意大小寫和形狀（O 和 0、I 和 1、S 和 5、B 和 8 的形狀差異）
- 中文字：逐字辨認，注意筆劃完整性`;

// ── Output format (shared) ──

const OUTPUT_FORMAT = `
輸出格式：
返回一個 JSON 物件，包含 "fields" 陣列：
{ "fields": [ { "name": "requested field name, if provided", "label": "文件中印的標籤文字", "value": "你實際讀到的值", "confidence": "high|medium|low" } ] }
- confidence=high: 每個字元都清晰可辨
- confidence=medium: 部分字元有些模糊但可合理判斷
- confidence=low: 有字元無法確認，已用 [?] 標記`;

// ── Category-specific prompts ──

/** scan: printed text on scanned documents */
const SCAN_PROMPT = `你是高精度文件閱讀器。眼前是一份掃描文件，文字是印刷體。
${READING_METHOD}
${ANTI_HALLUCINATION}
${OUTPUT_FORMAT}`;

/** photo: camera-captured document photos */
const PHOTO_PROMPT = `你是高精度文件閱讀器。眼前是一張用相機拍攝的文件照片。
${READING_METHOD}

照片特有注意事項：
- 照片可能有透視變形、光線不均、陰影。這些不影響文字內容，只影響辨識難度
- 光線不足的區域：更仔細地逐字確認
- 陰影覆蓋的文字：如能辨認就讀取，不能則標記 [?]

${ANTI_HALLUCINATION}
- 額外：不要因為照片品質差就推測內容，品質差只意味著更多 [?]
${OUTPUT_FORMAT}`;

/** handwriting / mixed: handwritten or mixed documents */
const HANDWRITING_PROMPT = `你是高精度文件閱讀器。眼前是一份包含手寫文字的文件。
${READING_METHOD}

手寫文字特有注意事項：
- 手寫文字：逐筆劃追蹤，從左到右逐字讀取
- 連筆字：追蹤每個筆劃的起點和終點，不要因為連筆而合併字元
- 印刷字和手寫字混合時：分別處理，不互相參考

${ANTI_HALLUCINATION}
- 額外：手寫字的辨識難度更高，不確定的字元應更積極地標記 [?]
- 不要因為前後文推測手寫內容（例如不能因為上面寫了「金額」就假設下面是數字）
${OUTPUT_FORMAT}`;

/** table: structured table extraction (kept for backward compat, returns scan prompt) */
const TABLE_PROMPT = SCAN_PROMPT;

// ── Prompt cache ──

const promptCache = new Map<DocumentType, string>();
promptCache.set("scan", SCAN_PROMPT);
promptCache.set("photo", PHOTO_PROMPT);
promptCache.set("handwriting", HANDWRITING_PROMPT);
promptCache.set("mixed", HANDWRITING_PROMPT);
promptCache.set("table", TABLE_PROMPT);

/**
 * Get the adaptive reading prompt for a document type.
 * Falls back to SCAN_PROMPT for unknown types.
 */
export function getReadingPrompt(docType: DocumentType): string {
  return promptCache.get(docType) || SCAN_PROMPT;
}

/**
 * Build a complete extraction prompt by combining:
 * 1. The adaptive reading prompt for this document type
 * 2. Optional field hints (label patterns, format hints only — NO spatial coordinates)
 * 3. Optional cross-page hint (multi-page template awareness)
 */
export function buildAdaptiveExtractionPrompt(
  docType: DocumentType,
  fieldLabels?: string[],
  crossPageHint?: string
): string {
  let prompt = getReadingPrompt(docType);

  if (crossPageHint) {
    prompt = prompt.replace(
      "眼前是一份",
      `眼前是一份${crossPageHint}`
    );
  }

  if (fieldLabels && fieldLabels.length > 0) {
    const labelList = fieldLabels.map((l) => `"${l}"`).join("、");
    prompt += `\n\n此文件預期包含以下欄位標籤（僅供參考，仍以文件實際內容為準）：${labelList}。`;
    prompt += `\n如果某個標籤在文件中找不到對應的值，該欄位的 value 留空 ""。`;
  }

  return prompt;
}

export interface FieldPromptSpec {
  name: string;
  labelPattern: string;
  formatHint?: string;
  example?: string;
  allowedValues?: string[];
  contextRule?: string;
  required?: boolean;
}

/**
 * Field-aware prompt for structured extraction.
 *
 * The older prompt only listed labels, so the model often returned nearby
 * values such as charge amounts for charge-code fields. This prompt gives the
 * model the requested output key and the value shape while still forbidding
 * hallucinated corrections.
 */
export function buildFieldExtractionPrompt(
  docType: DocumentType,
  fieldSpecs: FieldPromptSpec[],
  crossPageHint?: string
): string {
  let prompt = getReadingPrompt(docType);

  if (crossPageHint) {
    prompt = prompt.replace(
      "眼前是一份",
      `眼前是一份${crossPageHint}`
    );
  }

  const lines = fieldSpecs.map((f) => {
    const parts = [
      `name="${f.name}"`,
      `label_aliases="${f.labelPattern || f.name}"`,
    ];
    if (f.formatHint) parts.push(`return_value_shape="${f.formatHint}"`);
    if (f.example) parts.push(`example="${f.example}"`);
    if (f.allowedValues?.length) parts.push(`allowed_values="${f.allowedValues.join("|")}"`);
    if (f.contextRule) parts.push(`context_rule="${f.contextRule}"`);
    return `- ${parts.join("; ")}`;
  }).join("\n");

  prompt += `\n\n先完整讀取並保留整頁所有可見資料，再把以下 requested_fields 映射到最可能的欄位；不要因 requested_fields 而忽略其他欄位：\n${lines || "- 無 requested_fields：請自動發現所有欄位"}`;
  prompt += `\n\n欄位輸出規則：`;
  prompt += `\n- 每個輸出物件的 "name" 必須完全等於 requested_fields 的 name。`;
  prompt += `\n- "label" 寫文件中實際看見的標籤；"value" 只寫該欄位要求的值，不要寫解釋。`;
  prompt += `\n- 根據 requested field 的 label、format hint 和文件可見上下文讀取對應值；不要把鄰近欄位或標籤文字混入 value。`;
  prompt += `\n- 對所有字母與數字逐字閱讀；特別區分 O/0、I/1/L、S/5、B/8、M/N、U/V。`;
  prompt += `\n- 有任何字元不確定時，confidence 不能是 high，並用 [?] 標示不確定字元。`;
  prompt += `\n\n請返回 JSON，格式必須是：`;
  prompt += `\n{ "fields": [ { "name": "requested_name", "label": "visible label", "value": "exact visible value", "confidence": "high|medium|low" } ] }`;

  return prompt;
}

/**
 * v11 lossless prompt.
 * The model must preserve all visible content first, then map requested fields
 * as a derived view. Unknown labels and orphan values are first-class output.
 */
export function buildLosslessDocumentPrompt(
  docType: DocumentType,
  fieldSpecs: FieldPromptSpec[] = [],
  pageNumber?: number
): string {
  const requested = fieldSpecs.map((f) => {
    const parts = [
      `name="${f.name}"`,
      `label_aliases="${f.labelPattern || f.name}"`,
    ];
    if (f.formatHint) parts.push(`return_value_shape="${f.formatHint}"`);
    if (f.example) parts.push(`example="${f.example}"`);
    if (f.allowedValues?.length) parts.push(`allowed_values="${f.allowedValues.join("|")}"`);
    if (f.contextRule) parts.push(`context_rule="${f.contextRule}"`);
    return `- ${parts.join("; ")}`;
  }).join("\n");

  return `${getReadingPrompt(docType)}

你現在執行 lossless_document_v1 文件解析。核心要求：
- 必須保留整頁所有可見文字、數字、表格、標籤、未知欄位和孤立數據。
- requested_fields 只用於把已讀到的內容映射到指定欄位；不能因為 requested_fields 未列出而丟棄其他資料。
- 無法判斷欄位名稱時，保留到 unmapped_fields；只有值沒有明確標籤時，保留到 orphan_values。
- 模糊字元使用 [?]，不要根據文件類型或上下文補全。
- bbox 使用 Qwen 0-999 normalized coordinates，格式為 { "x1": number, "y1": number, "x2": number, "y2": number }；看不準 bbox 可以省略，但文字和值不能省略。
- 表格逐列逐格保留；無法判斷表頭時也要保留 cell 文字。

requested_fields:
${requested || "- 無。請自動發現所有欄位，不要輸出空殼欄位。"}

返回有效 JSON。JSON keyword required。格式必須是：
{
  "schema": "lossless_document_v1",
  "pages": [{
    "page": ${pageNumber ?? 1},
    "raw_markdown": "整頁文字與表格的 markdown，保留換行與順序",
    "raw_html": "如能可靠保留版面可輸出，否則空字串",
    "text_items": [{ "text": "逐行或逐詞文字", "bbox": { "x1": 0, "y1": 0, "x2": 999, "y2": 999 }, "confidence": "high|medium|low", "source": "full_page" }],
    "tables": [{ "bbox": { "x1": 0, "y1": 0, "x2": 999, "y2": 999 }, "rows": [["cell text"]] }],
    "field_candidates": [{ "name": "推測欄位名或空字串", "label": "文件上實際標籤", "value": "實際看到的值", "bbox": { "x1": 0, "y1": 0, "x2": 999, "y2": 999 }, "confidence": "high|medium|low", "source": "full_page", "needs_review": false }],
    "mapped_fields": { "requested_name": { "value": "映射到 requested field 的值", "label": "visible label", "confidence": "high|medium|low", "candidates": [] } },
    "unmapped_fields": [{ "label": "未知或未要求的標籤", "value": "值", "confidence": "high|medium|low" }],
    "orphan_values": [{ "value": "沒有明確標籤但可見的數據", "bbox": { "x1": 0, "y1": 0, "x2": 999, "y2": 999 }, "confidence": "high|medium|low" }],
    "uncertain_tokens": [{ "text": "[?]", "context": "周邊文字", "bbox": { "x1": 0, "y1": 0, "x2": 999, "y2": 999 } }]
  }]
}`;
}
