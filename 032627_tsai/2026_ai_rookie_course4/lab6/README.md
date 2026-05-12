# Lab6：模板消融實驗與交付封裝

## 目的

1. **對照實驗**：同一批測試資料，一種用**正確的 chat template**（跟 Lab5 一樣），一種用**亂拼的 prompt**（故意不用 template），再用跟 Lab5 一樣的 `evaluate_one` 比分數跟輸出品質。  
2. **交付**：生出給別人用的 **`inference.py`**（可執行骨架）和 **`README_delivery.txt`**，寫清楚模型、adapter 路徑、怎麼跑。

## 學習目標

- 具體觀察「**訓練／推理 template 不一致**」或「**不用 chat template**」對輸出與規則分數的影響。
- 練習把推理流程收斂成 **CLI 參數**（`--model_id`、`--adapter_dir`、`--user` 等）與簡短交付說明。

## 前置需求

- 在專案根目錄執行 `uv sync`。
- 與 **Lab5** 相同：基礎模型、4-bit 環境、`workdir/adapter`、`workdir/test.jsonl`。
- `lab6.py` 開頭有 `from ..lab5.lab5 import evaluate_one`，必須以**模組方式**從專案根目錄執行（已附 `lab5/__init__.py`、`lab6/__init__.py`）。請在根目錄執行：
  ```bash
  uv run python -m lab6.lab6
  ```
  若你直接 `uv run python lab6/lab6.py` 出現 **ImportError**，請改用上列指令（`main` 內已以腳本所在目錄解析 `lab6/workdir`，從根目錄執行時路徑仍正確）。另可擇一處理：**將 `evaluate_one` 複製到 `lab6.py`**，或調整 `sys.path` 後改寫 import，並在報告中註明。

## 建議實作順序（對照 `lab6.py`）

### TODO 1 — `generate_correct`

- 與 Lab5 的 `generate_reply` 邏輯相同：`apply_chat_template(..., add_generation_prompt=True)` → tokenize → `generate` → 解碼回覆。

### TODO 2 — `generate_wrong`

- **不要**使用 `apply_chat_template`。  
- 將 `messages` 內所有 **`user` 的 content** 串成一段 `user_text`（多則可用換行連接）。  
- 自訂簡單 prompt，例如：  
  `請用繁體中文回答：\n{user_text}\n回答：`  
  再 tokenize、generate。此作法模擬「以為只要拼字串就好」的常見錯誤。

### `run_template_ablation`（已給邏輯）

- 對前 `max_samples` 筆印出正確／錯誤模板的分數與截斷後的回覆，方便口頭報告。

### TODO 3 — `write_inference_script`

- 將完整可執行的**推理腳本內容**寫入 `path`（預設 `workdir/inference.py`）。  
- 建議包含：`argparse`、`load` base + Peft、`--system` / `--user` 組 `messages`、打印模型回覆。  
- 骨架內 `code = """..."""` 僅為占位；請替換成實際程式碼字串（注意多行字串跳脫）。

### TODO 4 — `write_readme`

- 可依需求擴充 `readme` 字串：硬體需求、Python 版本、`uv sync`／`uv run` 範例指令、常見錯誤。

### TODO 5～6 — `main`

- 呼叫 `run_template_ablation`；再呼叫 `write_inference_script` 與 `write_readme`。

## 選做：packing True / False

- 若課程要比 **packing**，就在 **Lab4** 跑兩次訓練，只改 `SFTConfig` 的 packing（其他盡量一樣），記 **loss 曲線或最後 loss**；這裡只要交對比結論，附圖或數字即可。

## 執行

在**專案根目錄**（與 `pyproject.toml` 同層）：

```bash
uv run python -m lab6.lab6
```

若 import 失敗，請依上一節「前置需求」調整後再執行。

## 完成定義

- 終端機可看到多筆「正確模板 vs 錯誤模板」分數對照，且 `workdir/inference.py`、`workdir/README_delivery.txt` 已寫入實質內容（非僅 TODO 占位）。  
- 簡短書面結論：**錯誤模板**在哪些評分項（禮貌、結構、中文）上明顯較差。
