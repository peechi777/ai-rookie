# Lab4：指令微調（SFT + PEFT / LoRA）

## 目的

拿 **Lab3** 的對話資料，用 **TRL** 的 `SFTTrainer` 加上 **PEFT（LoRA）** 對小型聊天模型做一輪**短**指令微調，把 **adapter 跟 tokenizer** 存到指定資料夾，給 Lab5 載入推理。

> 講義若提到 **QLoRA（4-bit）**、**packing**，這支範例程式預設是 **bfloat16 全精度 + LoRA**；顯存與 `trl` 版本允許的話，可自行在 `SFTConfig` 加 `load_in_4bit`、`dataset_text_field`、`packing` 等（並同步改 `from_pretrained`、`BitsAndBytesConfig`）。最低標準是**把 TODO 補完、訓練能跑完**。

## 前置需求

- 在專案根目錄執行 `uv sync`（已含 `torch`、`transformers`、`datasets`、`trl`、`peft`、`accelerate`、`bitsandbytes`）。
- GPU 建議 8GB 以上（`TinyLlama-1.1B` + LoRA 較輕量；若仍 OOM，請降 `per_device_train_batch_size`、`max_length` 或改 4-bit）。

## 資料準備

1. 確認 **Lab3** 已產生 `lab3/train.json`（或你修改後的檔名），內容為**物件列表**，每筆至少含 **`messages`**。
2. `lab4.py` 中 `load_dataset("json", data_files="../lab3/train.json")` 路徑為相對 **lab4 目錄**；若你從其他位置執行，請改成正確路徑。
3. 若有 **validation** 集，可擴充 `DatasetDict` 並在 `SFTTrainer` 傳入 `eval_dataset`（選做）。

## 建議實作順序（對照 `lab4.py`）

### TODO 1 — `formatting_samples`

- 使用 `tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)` 得到單一字串 `text`。
- **重點**：`SFTTrainer` 需要資料集提供訓練用欄位。依你使用的 **TRL 版本**，常見做法為回傳 `{"text": text}`，或在較新版本使用 `SFTConfig` 的 `dataset_text_field="text"`。若目前 `map` 回傳的是 token 字典，請對照官方文件改為與 `SFTTrainer` 相容的格式（錯誤訊息若提示缺少 `text` 或欄位不符，請以 `text` 欄位對齊）。
- 移除 `remove_columns` 後，應只剩模型訓練需要的欄位。

### TODO 2

- 已示範載入 train；請依實際檔案補上 val（選做）。

### TODO 3

- `ds.map(...)` 的 lambda 應呼叫 `formatting_samples`，使每筆變成模型可吃的格式。

### TODO 4 — 模型與 gradient checkpointing

- 骨架已 `gradient_checkpointing_enable()` 並設 `use_cache = False`（與 checkpointing 搭配時常見）；若之後改為推理腳本，記得區分訓練／推理設定。

### TODO 5 — `LoraConfig.target_modules`

- 針對 **TinyLlama（LLaMA 類）**，常見目標為 attention 與 MLP 的線性層，例如：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`（實際名稱可用 `print(model)` 查閱）。
- 部分環境可使用 PEFT 的簡寫（視版本而定）；若不清楚，以上列名稱逐一列出最穩。

### TODO 6～7 — `SFTConfig` 與 `SFTTrainer`

- 骨架已給 `output_dir`、`epochs`、`batch`、`learning_rate` 等；可依課堂要求調整 **`max_steps`**（短跑實驗可設小一點以免太久）。
- 建立 `SFTTrainer(model=peft_model, train_dataset=..., args=sft_cfg, ...)`；若資料欄位為 `text`，確認 `SFTConfig` 有對應設定。
- 呼叫 `trainer.train()`。

### TODO 8 — 儲存

- `trainer.save_model("adapter")` 與 `tokenizer.save_pretrained("adapter")` 將產出 **Lab5** 需要的目錄（若 Lab5 使用 `workdir/adapter`，請將輸出複製或改路徑一致）。

## 執行

```bash
cd lab4
uv run python lab4.py
```

訓練過程會寫 log；完成後應在 `lab4/adapter`（或你設定的 `output_dir`）看到 adapter 與 tokenizer 檔案。

## 自我檢核

- 訓練可跑完無崩潰；`adapter` 目錄內有 `adapter_config.json` 與權重檔。
- 能說出你改的 **learning rate、batch、LoRA rank** 對訓練穩定度與速度的影響（簡短即可）。

## 常見問題

| 現象 | 處理方向 |
|------|----------|
| `Field required: text` 或 dataset 欄位錯誤 | 在 `map` 後保證有 `text` 欄位，並設 `dataset_text_field="text"`。 |
| OOM | 降 `max_length`、`per_device_train_batch_size`，或開 4-bit + QLoRA。 |
| loss 不下降 | 檢查 template 是否 `add_generation_prompt=False`、資料是否為繁中與單一格式。 |

## 完成定義

- 成功完成至少一段短訓練並儲存 LoRA；實驗紀錄含主要超參數與資料來源。
