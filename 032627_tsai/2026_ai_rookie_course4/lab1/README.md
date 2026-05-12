# Lab1：Chat Template 轉換與一致性檢查

## 目的

微調跟推理時，模型吃的通常不是純 user 字串，而是經過 **chat template** 排好的字串（含 special tokens、角色標記）。這關要你：

- 讀 **JSON**（每筆有 `id`、`messages`，message 有 `role`、`content`）。
- 用 `AutoTokenizer.apply_chat_template` 轉成訓練／前處理用的文字。
- 寫簡單 **一致性檢查**，避免訓練、推理格式不一致或 messages 結構／順序有誤。

> **檔名對照**：`lab1.py` 檔頭註解寫 `lab2_chat_template.py`，內容就是 **Chat Template**；照這份說明和 `lab1.py` 裡的 `TODO` 做即可。

## 學習目標

- 能正確區分：`add_generation_prompt=False`（訓練／資料前處理）與 `True`（要接 `model.generate` 時）的差異。
- 能補齊或保留 **system** 角色，使資料格式穩定。
- 能對產生的字串做啟發式檢查，列出 `issues` 供除錯。

## 建議步驟（對照 `lab1.py` 的 TODO）

1. **環境**  
   在專案根目錄已執行過 `uv sync`。需能連線下載 tokenizer：`BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`（僅載 tokenizer 即可，不必載完整 3B 權重做推理）。
2. **TODO 1 — `ensure_system_message`**  
   - 若 `messages` 第一則的 `role` 不是 `system`，請在**最前面**插入一則預設 system，例如：`"你是專業客服助理，請用繁體中文，語氣禮貌。"`  
   - 若已有 system，則不要重複插入。
3. **TODO 2～3 — `to_chat_template_text`**  
   - 從 `example["messages"]` 取出列表，先經 `ensure_system_message` 處理。  
   - 呼叫 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`，回傳字串。
4. **TODO 4～5 — `check_template_consistency`**  
   - 檢查 **資料是否完整**：每一則 message 皆為 dict，且具備 `role`、`content` 欄位。  
   - 檢查 **content 是否為空**：`content` 經去除首尾空白後不應為空字串；`role` 也不應為空白。  
   - 檢查 **role 順序**：第一則必須為 `system`；之後須為 `user` 與 `assistant` 交替（索引 1、3、5… 為 `user`，2、4、6… 為 `assistant`）。單一則 `system`＋`user` 這種結尾在 user 的情況視為合理。  
   - 函式簽名為 `check_template_consistency(chat_text, messages)`；`messages` 請傳入與 `apply_chat_template` 相同的那份列表（`main` 已用 `ensure_system_message` 示範）。  
   - 回傳 `{"issues": [...], "length": len(chat_text)}`。

## 執行與自我檢核

```bash
cd lab1
uv run python lab1.py
```

- 對 `RAW_EXAMPLES` 裡的 `ex1`、`ex2` 都應印出長度與 `issues`。  
- `ex2` 刻意沒有 system：實作正確時，模板結果應仍帶有你補上的 system 導引；傳入檢查的 `messages` 在補上 system 後，`issues` 應反映結構與順序是否正確（範例資料預期可為空列表）。

## 與後續 Lab 的關係

- **Lab3** 的 `messages`、**Lab4** 的 `formatting_samples`、**Lab5** 推理時的 `apply_chat_template`，最好跟這裡講的格式一致（同一模型族，或刻意對齊 template）。

## 完成定義

- `ensure_system_message`、`to_chat_template_text`、`check_template_consistency` 實作完成，`uv run python lab1.py` 可跑完且輸出合理。

## `apply_chat_template` 簡短示範

把 `messages`（每則含 `role`、`content`）轉成模型要吃的字串。下面範例跟 `lab1.py` 一樣用 `Qwen/Qwen2.5-3B-Instruct`：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
messages = [
    {"role": "system", "content": "你是專業客服助理，請用繁體中文。"},
    {"role": "user", "content": "我想取消訂單。"},
]

# 訓練／前處理：回傳字串，不要加「下一輪 assistant」開頭
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)

# 若要接 model.generate，把上面改成 add_generation_prompt=True
```

`tokenize=False` 表示回傳文字；若要直接拿 tensor，可改 `tokenize=True` 並加上 `return_tensors="pt"`。
