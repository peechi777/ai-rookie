# Lab0：環境檢查與第一次生成

## 目的

先確認電腦能從 Hugging Face 載入因果語言模型，用 **Chat Template** 組好提示詞並跑完一次 `generate`。環境沒問題，後面幾關才不會一直踩雷。

## 為什麼本課程用 uv？

**uv** 是現代 Python 的套件與環境管理工具（由 Astral 維護），在本專案中我們用它取代「自己建 venv + `pip install -r requirements.txt`」的典型流程。對教學與實務的好處包括：

1. **速度快**：解析依賴與安裝套件以 Rust 實作，在課堂上同步安裝時等待時間明顯縮短。
2. **一處宣告**：依賴集中在專案根目錄的 `pyproject.toml`，版本與套件名稱好查、好改，也方便與其他工具整合。
3. **`uv run` 省去 activate**：不必每次 `source .venv/bin/activate`（或 Windows 的 `Scripts\activate`）；`uv run` 會自動使用專案對應的虛擬環境執行指令，減少「裝在系統 Python、跑在別套環境」的錯誤。
4. **可重現環境（選用）**：執行 `uv lock` 可產生 `uv.lock`，將鎖定的版本納入版控後，助教與同學之間較容易得到一致環境。
5. **與 pip／PyPI 相容**：語意仍是「從 PyPI 裝套件」，學會 uv 不影響你閱讀一般 Python 專案或文件裡的 `pip` 說明。

**安裝 uv**：請參考官方文件 [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)。

## 學習目標

1. 確認 **PyTorch** 版本與 **CUDA / GPU** 是否可用（若無 GPU，程式仍應能以 CPU 跑通，只是較慢）。
2. 成功用 `AutoTokenizer` / `AutoModelForCausalLM` 載入預設基礎模型（見 `lab0.py` 內 `BASE_MODEL_ID`，可用環境變數 `BASE_MODEL_ID` 覆寫）。
3. 將 `messages`（`system` + `user`）透過 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 轉成字串，再 tokenize 後呼叫 `model.generate`，並解碼得到模型回覆。

## 建議步驟（對照 `lab0.py`）

1. **同步依賴（在專案根目錄，與 `pyproject.toml` 同層）**  
   第一次請先執行：
   ```bash
   uv sync
   ```
   這會依 `pyproject.toml` 建立／更新 `.venv` 並安裝課程所需套件（含 `torch`、`transformers` 及後續 Lab 會用到的 `bitsandbytes`、`peft` 等）。若需固定全班相同版本，可在根目錄另外執行 `uv lock` 並將產生的 `uv.lock` 一併使用。
2. **執行腳本**  
   在 `lab0` 目錄下：
   ```bash
   cd lab0
   uv run python lab0.py
   ```
   （`uv` 會往上一層找到根目錄的 `pyproject.toml`，因此也可在根目錄執行：`uv run python lab0/lab0.py`。）
3. **自我檢核**
   - 終端機應印出：`PyTorch` 版本、`GPU 可用: True/False`（無 GPU 時為 `False` 屬正常）。
   - 應看到 `Chat template 文字:` 後面是一段依模型而定的對話格式字串。
   - 最後應有 `模型回應:` 與一段生成文字（內容每次可能不同，因有 `do_sample=True`）。

## 常見問題

| 現象 | 可能原因與處理 |
|------|----------------|
| 下載模型很慢或失敗 | 第一次會從 Hugging Face 拉權重；可設定鏡像、代理，或改用較小模型並設 `BASE_MODEL_ID`。 |
| `CUDA out of memory` | 換更小模型，或確認未同時開其他佔用顯存的程式；Lab0 的 `TinyLlama` 通常負擔較小。 |
| `apply_chat_template` 報錯 | 確認 tokenizer 來自 **Chat** 或 **Instruct** 類模型；純 base 模型可能沒有 chat template。 |

## 完成定義

- `uv run python lab0.py`（於 `lab0` 目錄）可無錯誤跑完，且能看到一次完整的「模板 → generate → decode」輸出。
