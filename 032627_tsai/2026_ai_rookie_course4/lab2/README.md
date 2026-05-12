# Lab2：Tokenizer 與訓練預算估算

## 目的

不同模型的 **Tokenizer** 對同一段中文、英文、程式碼會切出不同 token 數，牽涉 **訓練成本、上下文長度、計費**。這關要做兩件事：

1. 對多個模型（`lab2.py` 的 `CANDIDATE_TOKENIZERS`）算樣本的 **raw token**，以及 **套上 chat template 後**的 token 數。
2. 寫一個簡單的 **訓練預算估算**：依筆數、平均 prompt／回覆長度、epoch、假設的每秒 token 處理量，粗估總 token 跟大概要跑多久。

> **檔名對照**：`lab2.py` 檔頭註解寫 `lab1_tokenizer_and_budget.py`，內容是 **Tokenizer 與預算**；照這份說明和 `lab2.py` 的 `TODO` 做即可。

## 學習目標

- 理解「同樣字數，不同模型 token 數可能差很多」。
- 理解 **SFT 一筆樣本** 常同時包含 prompt（多輪 messages）與 assistant 回覆，估算時要拆成可解釋的變數。
- 能寫出可重用的 `token_count_report` 與 `estimate_training_budget`。

## 建議步驟（對照 `lab2.py`）

### TODO 1 — `token_count_report`

對每個 `model_id`：

1. `AutoTokenizer.from_pretrained(model_id, use_fast=True)`（若失敗可 `print` 警告並 `continue`，與骨架一致）。
2. 對 `SAMPLES` 裡每個 `(名稱, 文字)`：
   - **raw_tokens**：`len(tok.encode(text))` 或等價寫法。
   - **chat_tokens**：
     - 組裝 `messages`，例如：  
       `[{"role": "system", "content": "你是助理。"}, {"role": "user", "content": text}]`  
       （system 文案可自訂，但全專案建議與客服情境一致。）
     - `s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
     - `chat_tokens = len(tok.encode(s))`

回傳巢狀 dict，結構見函式 docstring。

### TODO 2～4 — `estimate_training_budget`

依課堂公式實作（與註解一致即可）：

- `total_tokens = num_samples * (avg_prompt_tokens + avg_resp_tokens) * epochs`
- `train_seconds = total_tokens / tokens_per_sec`
- `train_hours = train_seconds / 3600`

`tokens_per_sec` 為**粗估假設**（不同硬體、是否 packing、模型大小都會變），重點是學會數量級與敏感度。

### TODO 5 — `main` 內情境

骨架已示範一組假設；你可改成符合自己專題的筆數與長度假設，並觀察輸出變化。

## 執行與自我檢核

```bash
cd lab2
uv run python lab2.py
```

- 應印出 JSON 格式的 `report`（每個模型、每種語言的 `raw_tokens` / `chat_tokens` 皆為整數）。
- 應印出 `預算估算` 含 `total_tokens`、`train_seconds`、`train_hours`。

## 延伸（選做）

- 在 `CANDIDATE_TOKENIZERS` 加入註解中的 `TinyLlama`、`Mistral` 等，比較同一繁中句子的 token 數差異（注意部分模型需權限或較大下載量）。

## 完成定義

- 所有 `TODO` 完成，無例外時輸出合理數字；刻意無法載入的模型僅跳過並警告。
