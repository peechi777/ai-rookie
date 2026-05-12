# Lab5：推理與批次評估

## 目的

把 **Lab4** 訓練好的 **LoRA adapter** 掛回**基礎模型**，對測試集一筆一筆生成回覆，再用**簡單規則**（禮貌、有沒有步驟結構、中文比例、跟主題有沒有關）打分，算平均分、統計錯誤類型，最後寫成 JSON 報告。

## 學習目標

- 熟練 **PeftModel.from_pretrained(base, adapter_dir)** 與 tokenizer 載入順序。
- 推理時 **`add_generation_prompt=True`**，與 Lab1／Lab4 訓練用模板區分清楚。
- 建立最陽春的 **offline 評估** 流程，為日後改用 BLEU、ROUGE、LLM-as-judge 打基礎。

## 前置需求

- 在專案根目錄執行 `uv sync`。
- 基礎模型：`lab5.py` 內 `BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`（需下載權重；顯存不足請向助教確認是否可改小模型並重訓 adapter）。
- 4-bit 載入：依賴 `bitsandbytes` 與相容的 CUDA 環境。
- **Adapter 路徑**：骨架為 `workdir/adapter`。請將 Lab4 產出的 adapter **複製**到 `lab5/workdir/adapter`，或修改程式中的 `adapter_dir`。
- **測試資料**：骨架讀取 `workdir/test.jsonl`（**每行一個 JSON**）。若 Lab3 只產生 `test.json`，請自行撰寫小腳本轉成 JSONL，或修改 `main` 讀取 JSON 列表。

## 建議實作順序（對照 `lab5.py`）

### `load_base_and_adapter`（已給骨架）

- 確認 `tokenizer` 從 **adapter 目錄**載入（與訓練時 template 一致）；若 adapter 內無 tokenizer，可 fallback 到 base（需自行確保與訓練一致）。

### TODO 1～2 — `generate_reply`

1. `prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
2. `inputs = tokenizer(prompt, return_tensors="pt").to(model.device)`（或 `model` 第一個參數裝置）
3. `outputs = model.generate(**inputs, max_new_tokens=..., do_sample=False 或 True, ...)`
4. **只解碼「新增」部分**較常見：可對 `outputs[0][inputs["input_ids"].shape[1]:]` 做 `decode`，避免把整段 prompt 當成模型回覆；若整段 decode 再裁切，請在報告中說明做法。

### TODO 3 — `evaluate_one`

- **polite**：是否含「您好」「請」等（可擴充列表）。
- **structured**：是否含「1.」「1)」「步驟」等與 Lab3 格式呼應的字樣。
- **zh_ok**：可用 `CJK_RE` 統計中文字元數與總字元數比例，超過門檻為 True。
- **topical**：是否含 `example` 的 `topic` 關鍵字，或 user 內容中的關鍵片段（規則可簡單，重點一致即可）。
- `score` 為四項布林平均（骨架已給）；`errors` 列出未通過項目的說明。

### TODO 4～6 — `main`

- 讀取所有 test 列，逐筆 `generate_reply` + `evaluate_one`。
- **avg_score**：所有 `score` 平均。
- **err_counter**：可對每筆的 `errors` 內字串做 `Counter` 累加。
- 寫入 `workdir/eval_results.json`（含每筆 `id`、`score`、`errors`、`reply` 等）。

## 目錄準備範例

在 `lab5` 下建立：

```text
lab5/
  workdir/
    adapter/       # 從 Lab4 複製
    test.jsonl     # 每行一筆，含 id、messages（與訓練相同結構）
```

`test.jsonl` 單行範例（僅示意）：

```json
{"id": "ex0001", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "topic": "退貨流程"}
```

## 執行

```bash
cd lab5
uv run python lab5.py
```

## 完成定義

- 能對整份測試集跑完推理與評估，終端機印出平均分與錯誤統計，且 `eval_results.json` 存在可打開檢查。

## 注意事項

- **基礎模型要跟訓練時同一個**：Lab4 用 TinyLlama、Lab5 卻用 Qwen，adapter **對不起來**；請統一模型，或只把這份練習當「程式填空」，報告裡說明你實際怎麼對齊。
