# E:\PDF(3)\2022.pdf — 第 1 頁完整資料提取

> **提取日期**: 2026-06-09  
> **MCP 工具**: vision-mcp / analyze_pdf  
> **策略**: concurrent（高解析度 + 手寫增強）  
> **耗時**: 154.5 秒  

---

## MCP 執行步驟記錄

| 步驟 | MCP 工具 | 參數 | 說明 |
|------|----------|------|------|
| 1 | `get_pdf_info` | `pdf_path="E:\PDF(3)\2022.pdf"` | 取得 PDF 基本資訊：55 頁，A4 (595×842 pt)，20.3 MB |
| 2 | `analyze_pdf` | `pages="1"`, `vl_high_resolution_images=true`, `prompt=(完整提取提示)` | 高解析度渲染第 1 頁 → 傳送至 Vision API 進行完整欄位 + 手寫字提取 |

### analyze_pdf Pipeline 配置
- **preprocessing**: true（圖像預處理增強）
- **handwriting**: true（手寫字識別）
- **enhanced_extraction**: true（增強提取）
- **thinking**: true（推理模式）
- **temperature**: 0（最確定性輸出）
- **concurrency**: 20

---

## 第 1 頁提取資料（Markdown 表格）

### A. 文件表頭 / 頂端區塊

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| 手寫標記 | 1G7-0 | 🔴 手寫 | 文件最頂端，黑色墨水手寫 |
| 公司標誌 | HAMBURG SÜD | 印刷 | 頂部中央，帶箭頭圖標 |
| 公司標語 | A Maersk Company | 印刷 | 公司標誌下方 |

### B. 右上角參考編號區塊

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| Page No | 1 | 印刷 | 右上角 |
| NUN | NUN/0010010798 | 印刷 | 右上角 |
| 參考編號 | 181021712\* | 印刷 | 右上角 |
| Client No | 80627165 | 印刷 | 右上角 |
| Date | 27.06.22 | 印刷 | 右上角 |

### C. 貨主 / 收貨人

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| Shipper / Consignee 貨主/收貨人 | ETK INTERNATIONAL LTD | 印刷 | 左側區塊 |
| 貨主地址 | 36 HENNESSEY ROAD | 印刷 | 左側區塊 |
| 貨主城市 | HONG KONG | 印刷 | 左側區塊 |

### D. 航運明細

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| Place of Receipt 收貨地點 | AUCKLAND METROPOLIT | 印刷 | |
| Vessel 船名 | NYK FUSHIMI | 印刷 | |
| Voyage No. 航次 | 113N | 印刷 | |
| Port of Loading 裝貨港 | TAURANGA, BOP, NZ | 印刷 | |
| Port of Discharge 卸貨港 | HONG KONG | 印刷 | |
| Place of Delivery 交貨地點 | HONG KONG | 印刷 | |
| B/L No. 提單號碼 | 507943 | 印刷 | |
| Reference 參照代碼 | 507943 | 印刷 | |
| Bill of Lading Number 提單編號 | AL031059A | 印刷 | |
| Office of Issue 簽發地點 | AUCKLAND NZ | 印刷 | |

### E. 貨品描述

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| Description of Goods 貨品明細 | STK/BK 1300 | 印刷 | |
| 貨品描述 | 20kg ABC BAGS Fresh Brown Onions | 印刷 | |
| HS Code | 070310 | 印刷 | |
| 貨品附加描述 | AND MORE | 印刷 | |
| B/L TOTAL WT/M | 26000.000 KGS | 印刷 | |

### F. 貨櫃資料

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| CONTAINER SEAL NOS. | (1-3) TAKE SIZE PACKS | 印刷 | |
| 容器詳細資料 | MBNU0252813 SA1839887 4570 40'RH 1300 | 印刷 | 貨櫃號碼/封條/尺寸 |

### G. 費用明細

| 收費代碼 | 描述 | 費用類型 | 幣別 | Extended Value | Amount |
|----------|------|----------|------|----------------|--------|
| 9203 | THC DESTINATION | -LUMP SUM- | HKD | 3800.00 | 3800.00 |
| 47 | DOC.FEE/5/L 15S | -LUMP SUM- | HKD | 550.00 | 550.00 |

### H. 匯率 / 稅率

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| RATES OF EXCHANGE | FROM HKD TO USD | 印刷 | |
| VAT % | 0.12866 | 印刷 | |
| Rate Applicability Date | （空白） | 印刷 | 空白欄位 |

### I. 收款 / 印章

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| 收款印章 | Hamburg Sud 06-Jul-2022 Received TT payment | 🟠 印章 | 「06-Jul-2022」為手寫日期，其餘為預印文字 |
| 手寫日期 | 06-Jul-2022 | 🔴 手寫 | 印章內手寫日期 |

### J. 底部支付區塊

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| CASH HK$ | （空白） | 印刷 | 空白 |
| HKD CHQ# | （空白） | 印刷 | 空白 |
| USD CHQ# | （空白） | 印刷 | 空白 |
| Received By: | （空白） | 印刷 | 空白 |
| IN OUR FAVOUR (HKD) | 4350.00 | 印刷 | |
| IN OUR FAVOUR (USD) | 559.67 | 印刷 | |
| VAT | 請查稅 | 印刷 | |
| Payment conditions 付款條件 | （空白） | 印刷 | 空白 |
| Penalty interest 罰則 | （空白） | 印刷 | 空白 |

### K. 公司 / 銀行資訊

| 欄位名稱 | 內容/數值 | 類型 | 備註 |
|----------|----------|------|------|
| 公司名稱 | Hamburg Sued Hong Kong Ltd | 印刷 | |
| 公司地址 1 | 23/F, Tower 3, Enterprise Financial Centre | 印刷 | |
| 公司地址 2 | 23/F, 111-113 Hing Fong Road | 印刷 | |
| 公司城市 | Hong Kong | 印刷 | |
| 電話 | Tel: (852) 2181 2222 | 印刷 | |
| 傳真 | Fax: (852) 2181 2132 | 印刷 | |
| 服務中心 | Service Centre, 111 Hing Fong Road | 印刷 | |
| 銀行 | Bank: HSBC, Hong Kong | 印刷 | |
| 帳號 | Account No.: 800-311480-001 | 印刷 | |
| Swift Code | HSBCHKHH | 印刷 | |
| 銀行地址 | 1 Queen's Road Central, Hong Kong | 印刷 | |
| 收款人名義 | For A/C of: HAMBURG SÜD | 印刷 | |

---

## 類型圖例

| 標記 | 說明 |
|------|------|
| 印刷 | 預印/打印文字 |
| 🔴 手寫 | 手寫字（黑色墨水） |
| 🟠 印章 | 公司印章（部分手寫日期） |

---

## 摘要

- **文件類型**: Hamburg Süd 海運提單 / 費用明細 (Bill of Lading / Freight Invoice)
- **貨主**: ETK INTERNATIONAL LTD, Hong Kong
- **船名/航次**: NYK FUSHIMI / 113N
- **裝貨港**: TAURANGA, BOP, NZ
- **卸貨港**: HONG KONG
- **提單號碼**: AL031059A
- **貨品**: 20kg ABC BAGS Fresh Brown Onions (HS Code: 070310)
- **總費用**: HKD 4,350.00 (含 THC Destination HKD 3,800 + Doc Fee HKD 550)
- **付款狀態**: 2022年7月6日 TT 付款已收
- **手寫內容**: 頂端「1G7-0」+ 印章內日期「06-Jul-2022」
