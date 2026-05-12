# Lab3：資料清洗與小型 SFT 資料集建立

## 目的

做一份**小型、繁中客服風格**的指令微調資料：生出對話、套用固定回覆格式、清洗去重、用 token／長度過濾，再切成 **train / test** 存檔，給 **Lab4** 訓練、**Lab5** 評估用。

## 學習目標

- 熟悉 SFT 常見樣本結構：`messages`（多輪 `role` + `content`）、metadata（如 `topic`、`language`）。
- 實作一條可講給別人聽的 **資料清洗管線**：繁化、正規化、毒性、長度、近似重複、train/test 切分。
- 了解「客服語氣＋步驟化」可用**固定模板**先做出可訓練資料，再視需要接 API 擴充。

## 前置需求

- 在專案根目錄執行 `uv sync`（已含 `transformers`、`tqdm`、`requests`、`opencc-python-reimplemented` 等）。若 OpenCC 匯入失敗，`lab3.py` 會 fallback 成不轉換（`to_trad` 原樣返回），請在報告中註明。
- （選用）腳本內有 `call_llm`，會對 `http://127.0.0.1:8299/v1/chat/completions` 發送請求。若你**沒有**在本機起相容 OpenAI Chat API 的服務，請以 **模板生成** 完成 `synth_assistant`，不要呼叫 `call_llm`，以免程式卡住或報錯。

## 客服回覆格式（必須遵守）

助理（`assistant`）的回覆應符合下列結構（可把 `{topic}`、`{user_query}` 代入實際主題與使用者原文）：

```text
顧客您好:
感謝您聯繫我們關於{topic}的問題。針對您的詢問："{user_query}"，我們建議您按照以下步驟進行：
1. 步驟一....
2. 步驟二....
3. 步驟三....
如果您在操作過程中遇到任何困難，請隨時與我們聯繫。我們的客服團隊將竭誠為您提供協助。
祝您有美好的一天！
客服團隊 敬上
```

步驟內文可依主題客製，但**開頭稱謂、感謝句、編號步驟、結尾祝福與「客服團隊 敬上」**應保持一致性，方便 Lab5 用規則評估「結構化／禮貌」。

## 建議實作順序（對照 `lab3.py`）

### TODO 0

- 確認 OpenCC：`from opencc import OpenCC` 與 `OpenCC("s2t")`；失敗則使用 fallback。

### TODO 1 — `normalize_text`

1. 簡轉繁：`to_trad(s)`  
2. Unicode 正規化：例如 `unicodedata.normalize("NFKC", ...)`  
3. 壓縮多餘空白：可用 `re.sub(r"\s+", " ", ...).strip()`

### TODO 2 — `is_toxic`

- 若 `s` 含有 `BADWORDS` 任一詞，回傳 `True`，否則 `False`。

### TODO 3～4 — `build_synthetic_examples`

- `more_topics.py` 的 `more_topics` 為 `(topic, user_query)` 列表；迴圈內用 `random.choice(more_topics)` 取一組。
- **`synth_assistant(topic, user)`**：回傳符合上一節**固定格式**的繁體字串（可用 f-string 填入 topic 與 user）。
- 組裝 **`messages`**：至少一則 `user` 與一則 `assistant`；建議再加上與 Lab1 一致的 `system`（例如專業客服、繁體、禮貌）。
- 每筆 append 的 dict 需含：`id`、`messages`、`topic`、`language`（骨架用 `zh-Hant`）。

### TODO 6～9 — `clean_dataset`

對每個 `ex`：

1. 複製或重建 `messages`，對每則 `content` 做 `normalize_text`；若任一則 `is_toxic` 為真，**整筆丟棄**（`continue`）。
2. 合併所有 **user** 的 `content` 為 `user_concat`，檢查字元長度：過短／過長則丟棄（門檻可自訂，與參數 `max_user_len` 對齊）。
3. 用 **`tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)`** 得到字串，再 `len(tokenizer.encode(...))` 估計 **chat token 數**；若超過 `max_total_tokens` 則丟棄。
4. **去重**：對 `user_concat` 做 hash（例如 `hashlib.md5(user_concat.encode("utf-8")).hexdigest()`），若已出現在 `seen_keys` 則跳過，否則加入並保留該筆。
5. 通過者 append 到 `cleaned`。

### `main` 與輸出檔

- 骨架中 `build_synthetic_examples(n=1)` 僅供快速測試；正式作業請改為較大 `n`（例如 60～200），否則 train/test 可能過少。
- 目前程式將資料存成 **`train.json`**、**`test.json`**（JSON 陣列，indent=4）。**Lab4** 的 `load_dataset("json", data_files=...)` 可讀取 JSON 陣列；若你改存 **JSONL**（每行一筆），請一併修改 Lab4 的 `data_files` 與讀取方式。
- 若課程要求 **驗證集**，可在切分時多切出 `val`（例如 8:1:1）並另存檔。

## 執行

```bash
cd lab3
uv run python lab3.py
```

完成後目錄下應有 `train.json`、`test.json`，且終端機印出筆數。

## 與 Lab4 / Lab5 的銜接

- **Lab4** 預設從 `../lab3/train.json` 讀取；請確認路徑與欄位名稱含 **`messages`**。
- **Lab5** 範例程式讀取 **`workdir/test.jsonl`**。若你只在 Lab3 產生 `test.json`，請在課程指定目錄下將測試集**轉成 JSONL**（每行一個 JSON 物件），或修改 Lab5 讀取邏輯與路徑，並在實驗記錄中說明。

## 完成定義

- 清洗管線可跑通，產出具 **system/user/assistant**、格式一致、無明顯髒話與重複 user 的 train/test 檔。
- 能口頭或書面說明每一道過濾規則的目的與取捨。
