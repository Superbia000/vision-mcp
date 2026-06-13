# Vision-MCP v10 最佳實踐指南

## Tool 選擇

| 目的 | 使用 Tool | 說明 |
|---|---|---|
| 先看檔案狀態、頁數、token 估算 | `vision_inspect` | PDF/image/video 都用這個入口 |
| 預覽渲染或 OCR 前處理 | `vision_prepare` | 支援 scan/table/photo/handwriting/negative |
| 非結構化分析、全文 OCR、摘要、影片理解 | `vision_analyze` | 自動選 image/pdf/video pipeline |
| 表單、發票、提單、銀行文件欄位抽取 | `vision_extract` | full-page first，低信心欄位再驗證 |
| Batch/async job 提交與查詢 | `vision_jobs` | 只在 provider/region/model 支援時使用 |

## 準確度優先

```json
{
  "accuracy_mode": "max",
  "vl_high_resolution_images": true,
  "temperature": 0,
  "ocr_verify": true,
  "self_consistency_votes": 3
}
```

## 速度優先

```json
{
  "accuracy_mode": "fast",
  "vl_high_resolution_images": false,
  "self_consistency_votes": 1
}
```

## 欄位抽取規則

- 使用 `vision_extract`，不要再直接調用舊的 `extract_document_fields`。
- 每個欄位盡量提供 `name`、`label_pattern`、`format_hint`。
- 無法可靠識別的欄位會標記 `needs_review=true`；這是正確行為，不應改成猜測值。
- PDF 多頁欄位抽取用 `pages` 指定範圍；單頁圖片直接傳 `file_path`。

## 舊 Tool 遷移

完整對照見 `docs/TOOL_MIGRATION_V10.md`。
