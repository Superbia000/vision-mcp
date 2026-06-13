# vision-mcp 深度測試與單模型優化報告

測試時間：2026-06-13  
目標 PDF：`C:\Users\Guest!\Desktop\60865 DCS 3 WCH\WCHRN 26022277 - 食品批發 - 假單 - IO\2022.pdf`  
備份：`C:\Users\TANG CHIU MING\.agents\mcp\vision-mcp\.backups\vision-mcp-pre-test-opt-20260613-002224.zip`

## 1. 測試結論

- 已透過 MCP SDK stdio 實際調用 MCP，不是只調用內部函式。
- 全部 14 個 MCP tools 均已覆蓋，25 個測試步驟全部成功。
- 原始全矩陣總工具耗時：906.1 秒。
- PDF 資訊：20.58 MB，56 頁，所有頁面 595 x 842 pt。
- 最佳泛用 OCR 路徑（未改前）：`analyze_image` PDF bridge，42.48 秒，頁 1 marker hit 16/20。
- 欄位抽取原始瓶頸：`extract_document_fields(auto)` 107.59 秒，且 charge code / container / seal 容易錯。
- 優化後 `extract_document_fields(auto)`：61.14 秒；表格路由成功抽到 `1G7-0`、`181021712*`、`MNBU0252813`、`SAL839887` 等關鍵欄位。
- 以目視可確認欄位計算，優化後 `extract_document_fields(auto)` 達 17/17。第二個 charge code 左側被筆跡遮擋，不納入高可信 exact 分母。

## 2. MCP 調用步驟

實際執行腳本：`node deep-test-mcp-20260613.mjs`

MCP 連線流程：

1. 讀取 Codex config 的 `[mcp_servers.vision-mcp.env]` 環境變數，API key 不輸出。
2. 以 `StdioClientTransport` 啟動 `node dist/index.js`。
3. `client.connect(transport)` 完成 MCP initialize。
4. `tools/list` 讀取全部工具 schema。
5. 逐項 `tools/call`，每次記錄 tool name、arguments、timeout、elapsed_ms、輸出 JSON/text。
6. 將結果寫入 `deep-test-results-20260613/`。

## 3. Tools / 參數覆蓋

| Tool | 已覆蓋參數 |
|---|---|
| `analyze_pdf` | `pdf_path`, `pages`, `prompt`, `max_tokens`, `concurrency`, `image_quality`, `max_image_width`, `enable_thinking`, `thinking_budget`, `vl_high_resolution_images`, `max_pixels`, `min_pixels`, `strategy`, `chunk_size`, `fields`, `self_consistency_votes`, `temperature`, `top_p` |
| `analyze_image` | `image_path`, `prompt`, `max_tokens`, `max_image_width`, `image_quality`, `enable_thinking`, `thinking_budget`, `vl_high_resolution_images`, `max_pixels`, `min_pixels`, `temperature`, `top_p` |
| `analyze_video` | `video_path`, `prompt`, `max_tokens`, `enable_thinking`, `thinking_budget`, `fps`, `nframes`, `temperature`, `top_p` |
| `get_pdf_info` | `pdf_path` |
| `analyze_pdf_batch` | `pdf_path`, `pages`, `prompt`, `max_tokens`, `image_quality`, `enable_thinking`, `thinking_budget`, `temperature`, `top_p` |
| `get_batch_status` | `batch_id` |
| `get_batch_status_all` | `batch_ids` |
| `estimate_tokens` | `file_path`, `pages`, `max_image_width`, `fps` |
| `analyze_pdf_large` | `pdf_path`, `pages`, `prompt`, `max_tokens`, `concurrency`, `chunk_size`, `image_quality`, `max_image_width`, `max_pixels`, `min_pixels`, `enable_thinking`, `thinking_budget`, `use_batch`, `vl_high_resolution_images`, `temperature`, `top_p` |
| `analyze_video_chunked` | `video_path`, `prompt`, `max_tokens`, `enable_thinking`, `thinking_budget`, `fps`, `nframes`, `chunk_duration_sec`, `aggregate` |
| `ocr_enhance_image` | `image_path`, `mode`, `output_path` |
| `extract_document_fields` | `image_path`, `fields`, `use_ocr_model`, `enable_thinking`, `self_consistency_votes`, `preprocess`, `strategy` |
| `extract_with_verification` | `image_path`, `prompt`, `validation_rules`, `use_thinking`, `preprocess`, `strategy` |
| `ocr_handwriting` | `image_path`, `prompt`, `language_hint` |

## 4. 原始全矩陣速度

| Step | Tool | 耗時 |
|---|---:|---:|
| T01 | `get_pdf_info` | 0.06s |
| T02 | `estimate_tokens` | 3.58s |
| T03 auto/light/aggressive/handwriting | `ocr_enhance_image` | 2.37s-3.02s |
| T04 | `analyze_image` PDF bridge | 42.48s |
| T05 | `analyze_pdf` page 1 basic | 44.38s |
| T06 | `analyze_pdf` page 1 fields | 53.96s |
| T07 | `analyze_pdf` pages 1-3 concurrent | 41.45s |
| T08 | `analyze_pdf` pages 1-3 multi-image | 90.04s |
| T09 | `extract_document_fields(auto)` before optimization | 107.59s |
| T10 | `extract_document_fields(full-page)` before optimization | 98.56s |
| T11 | `extract_with_verification` | 65.84s |
| T12 | `ocr_handwriting` | 136.80s |
| T13 | `analyze_pdf_large` pages 1-3 realtime | 90.73s |
| T14 | `analyze_pdf_batch` submit pages 1-2 | 56.38s |
| T15/T16 | batch status | 0.22s / 0.21s |
| T17/T18 | video tools | 6.85s / 6.88s |
| T20/T21 | temperature 0 / 0.3 | 25.46s / 24.28s |

Batch note：batch submit 成功，狀態查詢時仍為 `validating` / `in_progress`，這是非同步 provider queue 行為。

## 5. 準確度觀察

頁 1 目視真值校正：

- 公司 / shipper 是 `ETAK INTERNATIONAL LTD`，不是舊報告的 `ETK INTERNATIONAL LTD`。
- `B/L Number` 是 `AL03080A`。
- container / seal 是 `MNBU0252813` / `SAL839887`。
- 第二個 charge code 左側被筆跡遮住，不列入高可信 exact 分母。

原始結果：

| 路徑 | 結果 |
|---|---|
| `analyze_image` PDF bridge | marker hit 16/20，漏手寫頂碼、container 等 |
| `analyze_pdf` basic | marker hit 14/20 |
| `extract_document_fields(auto)` | 舊計分 10/16；主要錯 charge code、container、seal、手寫頂碼 |
| `extract_with_verification` | 產生較多推測，`client_no`, `vessel`, `B/L`, `container` 均有錯 |

優化後 focused retest：

| 路徑 | 耗時 | 結果 |
|---|---:|---|
| `extract_document_fields(auto)` | 61.14s | 目視可確認欄位 17/17；舊腳本未校正真值顯示 16/18 |
| `extract_document_fields(full-page)` | 45.08s | 舊腳本 12/18；仍錯手寫頂碼、container/seal |
| `analyze_pdf(fields)` | 57.20s | 輸出在 nested `results[0].text`；人工讀取接近 table route 結果 |

## 6. 已實作優化

1. `src/extraction/prompts.ts`
   - 新增 field-aware extraction prompt。
   - 要求模型回傳 requested `name`，避免後處理用 label 猜欄位。
   - `charge_code` 欄位明確要求回傳左側代碼，不要回傳金額。
   - 對 container/seal/B/L/client/invoice 增加逐字辨識提醒。

2. `src/extraction/router.ts`
   - `matchField()` 先匹配 `name/key/field`，再匹配 label pattern。
   - 新增本地 `format_hint` 驗證，格式不合不標 high verified。
   - multi-pass 第二輪只針對低信心或格式不合欄位。
   - multi-pass 重用同一份本地 preprocess 結果，減少本地處理開銷。

3. `src/tools/extraction.ts`
   - full-page route 不再先做 generic preprocess，避免雙重預處理。

4. `src/preprocessing/pipeline.ts`
   - 表格偵測閾值從 `edgeRatio < 0.15` 放寬至 `< 0.30` 並排除 high-noise。
   - 這份 PDF 頁 1 從 `scan` 正確改判為 `table`。

5. `src/extraction/layout.ts`
   - 增加 bbox-tolerant JSON parse。
   - 當模型只把 `bbox` key 輸出壞掉時，移除 `bbox` 後保留欄位值，不再整包失敗。

## 7. 驗證

已執行：

```powershell
npm run build
node --check .\deep-test-mcp-20260613.mjs
node --check .\focused-retest-20260613.mjs
rg -n "U+FFFD replacement char or repeated question marks" <changed files>
```

結果：

- TypeScript build 通過。
- 測試腳本語法通過。
- 中文替換字元檢查無命中。
- focused retest 使用新 `dist/index.js` 重新啟動 MCP server 驗證。

## 8. 剩餘限制

- 第二個 charge code 原圖被筆跡遮擋，無法可靠宣稱 100% exact。
- 現有已啟動的 Codex Desktop MCP server process 仍可能載入舊 `dist`；需要重啟 Codex Desktop 或 MCP server 才會使用新 build。
- 對極細字的 container/seal，table route 明顯優於 full-page route；後續若要再提高，可加入同模型裁切驗證，但會增加 API calls。
