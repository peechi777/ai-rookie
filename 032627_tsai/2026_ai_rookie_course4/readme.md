# LLM訓練與資料基礎：Tokenizer、Chat Template、資料格式 + 指令微調-SFT（PEFT）與資料設計



## 課程表
| 時間    | 內容 |
| -------- | ------- |
| 9:00~11:00  | 理論講解    |
| 12:00~18:00    | 實作課程 Lab1~6   |


## 課程目標

完成本課程後，你應該能夠：

- 理解不同 Tokenizer、Chat Template 對 LLM 訓練與推理的影響。
- 設計並清洗一份用於指令微調（SFT）的繁體中文對話資料集。
- 使用 PEFT（LoRA/QLoRA）完成一次端到端的指令微調。
- 在推理階段正確套用 Chat Template，並對模型輸出做基本評估與錯誤分析。

### 環境與依賴（uv）

本專案以 **[uv](https://github.com/astral-sh/uv)** 管理 Python 版本與套件（取代手動 `venv` + `pip install -r requirements.txt`）。依賴宣告在專案根目錄的 `pyproject.toml`。

1. **安裝 uv**（擇一）：見官方文件 [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)。Windows 可先用獨立安裝程式或 `pip install uv`。
2. **在專案根目錄同步依賴**（會建立 `.venv` 並安裝 `pyproject.toml` 中的套件）：
   ```bash
   uv sync
   ```
3. **執行各 Lab 腳本**：在專案根目錄使用 `uv run`（會自動使用上述虛擬環境，無須先 `activate`）。各 Lab 的具體指令見各資料夾內 `README.md`。

> **GPU 與 PyTorch**：若需要特定 CUDA 版 PyTorch，請依 [PyTorch 官網](https://pytorch.org/) 或 uv 文件，以額外 index 安裝對應的 `torch` 後再執行 `uv sync`／調整 `pyproject.toml`。

**為什麼用 uv、細部說明與第一次產生鎖檔**：見 `lab0/README.md`。

## 實作 Lab 概觀

| Lab | 主題 | 目標 |
| --- | ---- | ---- |
| Lab0 | 環境檢查 | 確認 GPU/套件就緒，能載入模型並完成一次簡單對話生成。 |
| Lab1 | Chat Template 與訊息結構 | 使用 `apply_chat_template` 將對話資料轉為模型輸入，並檢查模板一致性。（對應資料夾 `lab1/`） |
| Lab2 | Tokenizer 與成本估算 | 比較不同模型的 token 數，建立訓練/推理預算估算器。（對應資料夾 `lab2/`） |
| Lab3 | 資料清洗與 SFT 資料集 | 建立並清洗小型繁中客服資料集，輸出 train/val/test JSONL。 |
| Lab4 | 指令微調（SFT + PEFT） | 使用 LoRA/QLoRA 對基礎模型進行短程指令微調。 |
| Lab5 | 推理與批次評估 | 載入微調後模型，批次推理並以簡單規則評估輸出品質。 |
| Lab6 | 消融實驗與打包交付 | 對比正確/錯誤模板效果，並產出推理腳本與 README。 |


