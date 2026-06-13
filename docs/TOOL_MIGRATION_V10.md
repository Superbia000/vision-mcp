# vision-mcp v10 Tool Migration Map

v10 將舊的 14 個 MCP tools 收斂成 5 個分類 tools。這是 breaking change：舊 tool 不再對外暴露。

## 對照表

| 舊 Tool | 新 Tool | 主要參數變化 |
|---|---|---|
| `get_pdf_info` | `vision_inspect` | `pdf_path` -> `file_path` |
| `estimate_tokens` | `vision_inspect` | `file_path` 保持不變 |
| `ocr_enhance_image` | `vision_prepare` | `image_path` -> `file_path` |
| `analyze_image` | `vision_analyze` | `image_path` -> `file_path` |
| `analyze_pdf` | `vision_analyze` | `pdf_path` -> `file_path` |
| `analyze_pdf_large` | `vision_analyze` | `pdf_path` -> `file_path`; use `strategy="chunked"` or `strategy="batch"` |
| `analyze_video` | `vision_analyze` | `video_path` -> `file_path` |
| `analyze_video_chunked` | `vision_analyze` | `video_path` -> `file_path`; use `strategy="chunked"` |
| `extract_document_fields` | `vision_extract` | `image_path` -> `file_path` |
| `extract_with_verification` | `vision_extract` | convert `prompt` intent into `fields` and `validation_rules` |
| `ocr_handwriting` | `vision_extract` | use `document_type="handwriting"` |
| `analyze_pdf_batch` | `vision_jobs` | `action="submit"`, `pdf_path` -> `file_path` |
| `get_batch_status` | `vision_jobs` | `action="status"`, `batch_id` -> `job_id` |
| `get_batch_status_all` | `vision_jobs` | `action="status_all"`, `batch_ids` -> `job_ids` |

## Examples

### PDF analysis

```json
{
  "tool": "vision_analyze",
  "arguments": {
    "file_path": "C:\\path\\document.pdf",
    "pages": "1-3",
    "prompt": "Extract all visible text and tables.",
    "accuracy_mode": "balanced"
  }
}
```

### Field extraction

```json
{
  "tool": "vision_extract",
  "arguments": {
    "file_path": "C:\\path\\invoice.pdf",
    "pages": "1",
    "accuracy_mode": "max",
    "fields": [
      { "name": "invoice_no", "label_pattern": "Invoice No|Invoice Number" },
      { "name": "total", "label_pattern": "Total|Amount", "format_hint": "currency" }
    ]
  }
}
```

### Batch submit

```json
{
  "tool": "vision_jobs",
  "arguments": {
    "action": "submit",
    "file_path": "C:\\path\\large.pdf",
    "pages": "1-200",
    "prompt": "Extract all text page by page.",
    "batch_policy": "auto"
  }
}
```
