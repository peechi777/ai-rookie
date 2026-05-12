# Lab 1：SFT 客服模型（約 2 小時）

## 情境

你拿到一份現成的客服對話資料集，要用它微調一個本地小模型。
任務很明確：**先量化 base model 有多爛，再訓練，再量化它進步了多少。**

## 資料集

[`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

- 27 個意圖（cancel_order, track_order, payment_issue...）
- 10 個大類
- 約 26,000 筆 instruction-response pairs
- 每個意圖約 1,000 筆樣本

## 練習模式說明

本 lab 採「填空式練習」：

- **Step 1 / Step 6** 已寫好，直接執行即可
- **Step 2 / Step 3 / Step 4 / Step 5** 是訓練流程的核心，需要你完成 `TODO`
- 每個 `TODO #N` 都附上**清楚的提示**：要做什麼、為什麼這樣做、有哪些常見參數
- 如果沒填 TODO 直接跑，程式會在那一行附近報錯（變數是 `None` 或 list 是空），知道該動哪裡

| Step | 是否需要填空 | TODO 數量 | 重點 |
|------|------------|----------|------|
| 1 | ❌ 直接執行 | 0 | 資料探索（觀察 intent 分布 / 回答長度） |
| 2 | ✅ 需要填空 | 2 | **分層抽樣**：每個 intent 獨立切 90/10 |
| 3 | ✅ 需要填空 | 2 | 評估流程：chat template + generate + ROUGE-L |
| 4 | ✅ 需要填空 | 4 | **LoRA config、format_chat、SFTConfig、save_pretrained** |
| 5 | ✅ 需要填空 | 1 | 用 `PeftModel` 接回 LoRA adapter |
| 6 | ❌ 直接執行 | 0 | Before vs After 對比報告 |

⚠️ **重要**：step3、step4、step5 的 `system prompt` 必須維持「開頭一致」，
　　否則 base model 評估與訓練後模型評估的條件不同，比較會失真。

## 操作流程

請在 `lab1/` 資料夾下，按順序執行以下步驟：

```bash
cd lab1
```

---

### Step 1：環境 + 資料探索（20 min）

```bash
uv run python step1_explore.py
```

**你會學到：**
- 載入 HuggingFace 資料集
- 觀察資料的欄位結構（category / intent / instruction / response）
- 分析意圖分布與回答長度

**觀察重點：**
- 資料集有幾種意圖？分布是否均勻？
- 回答的平均長度是多少？

---

### Step 2：切出評估集（10 min）

```bash
uv run python step2_split.py
```

**⚠️ 需要完成 2 個 TODO 才能執行：**

1. `TODO #1` — 用 `defaultdict(list)` 建立 `intent → [indices]` 對應表
2. `TODO #2` — 對每個 intent **獨立** 做 90/10 切分（這就是分層抽樣的精髓）

**你會學到：**
- 為什麼評估集必須在訓練前切好、鎖死
- 分層抽樣（stratified sampling）的概念：確保每個意圖在測試集都有樣本

**產出檔案：** `split_indices.json`（後續 step 會讀取）

---

### Step 3：訓練前評估 — Before（25 min）🔑

```bash
uv run python step3_eval_before.py
```

> **這是整個 Lab 最重要的環節之一。** 沒有 before 數據，after 的數字毫無意義。

**⚠️ 需要完成 2 個 TODO 才能執行：**

1. `TODO #1` — 把問題包成 system + user 兩輪 messages，呼叫 `model.generate(...)`
2. `TODO #2` — 用 `rouge_scorer` 計算 ROUGE-L 分數（取 `.fmeasure`）

**評估指標：**

| 指標 | 測什麼 | 怎麼算 |
|------|--------|--------|
| ROUGE-L | 回答與 ground truth 的文字重疊度 | 自動計算，客觀 |
| 回答長度比 | 模型是否廢話太多或太簡短 | `len(pred) / len(ref)` |
| 人工品質打分 | 回答是否真的有用 | 學生互評 1-5 分（抽 10 題） |

**你要做的事：**
1. 觀察 base model 的 ROUGE-L 分數
2. 觀察哪些意圖表現最差、哪些最好
3. **人工打分：** 程式會印出 10 題，請記錄你的 1-5 分評分

**產出檔案：** `eval_before.json`、`manual_eval_questions.json`

---

### Step 4：SFT 訓練（40 min）⭐ 訓練流程的主菜

```bash
uv run python step4_train.py
```

**⚠️ 需要完成 4 個 TODO 才能執行（這是整個 lab 的核心）：**

1. `TODO #1` — 建立 `LoraConfig`（r=16、alpha=16、target_modules、lora_dropout）
2. `TODO #2` — 在 `format_chat` 組 system / user / assistant 三輪對話
3. `TODO #3` — 建立 `SFTConfig`（epochs、batch、grad accum、max_steps、lr、max_length...）
4. `TODO #4` — 訓練完用 `model.save_pretrained("lab1_customer_support_adapter")` 存 adapter

**你會學到：**
- 用 Transformers 載入 BF16 模型
- 用 PEFT 加上 LoRA adapter（r=16）
- 用 TRL 的 SFTTrainer 進行 supervised fine-tuning
- 觀察 training loss 下降曲線

**關鍵設定（提示已寫在程式碼註解內）：**
- 訓練子集：6,000 筆（受 VRAM 限制）
- LoRA rank：16
- Max steps：600
- Learning rate：2e-4

**產出檔案：** `lab1_customer_support_adapter/`（LoRA adapter 權重）

**如果遇到 OOM：**
- 降 `per_device_train_batch_size` 到 2
- 降 LoRA `r` 到 8

---

### Step 5：訓練後評估 — After（20 min）🔑

```bash
uv run python step5_eval_after.py
```

**⚠️ 需要完成 1 個 TODO 才能執行：**

1. `TODO` — 用 `PeftModel.from_pretrained(base_model, "lab1_customer_support_adapter")` 把 step4 訓練好的 adapter 接回 base model

**陷阱：** 如果這行沒寫，後面整個 AFTER 評估與人工打分都會用 base model 跑，結果跟 step3 BEFORE 一樣，看不到訓練成效。

**你要做的事：**
1. 觀察 after 的 ROUGE-L 分數
2. **人工打分：** 程式會印出同樣的 10 題（before vs after），請記錄 after 的 1-5 分評分

**產出檔案：** `eval_after.json`

---

### Step 6：Before vs After 對比 + 討論（5 min）

```bash
uv run python step6_compare.py
```

**討論問題：**

1. ROUGE-L 提升了多少？這個數字有意義嗎？ROUGE 的 limitation 是什麼？
2. 哪些 intent 進步最大、哪些最小？為什麼？
3. 你的人工打分和 ROUGE 分數一致嗎？什麼情況下不一致？

