# Lab 2：Reasoning KD — 打 API 合成推理資料

## 情境

Lab 1 的 label 是預先寫的。現在你要做的是**真正的知識蒸餾**：
用一個強大的 Teacher 模型（GPT-4o-mini 或 DeepSeek）生成訓練資料——
而且不只生成答案，是生成**完整的推理過程**。

題目來源：GSM8K（小學數學），有標準答案可以客觀驗證。

## 為什麼這個設計好？

- **Teacher 是你自己打 API 叫來的** → 這才是 Black-box KD
- **推理資料 = CoT** → 直接對應投影片路線 2
- **GSM8K 有答案可以比對** → 不用人工評估，正確率一目了然
- **訓練前後差異巨大** → 1.5B 模型裸跑 GSM8K 通常不到 30%，蒸餾後可到 50%+

## 練習模式說明


- **Step 1 / Step 6** 已寫好，直接執行即可
- **Step 2 / Step 3 / Step 4 / Step 5** 是訓練流程的核心，需要你完成 `TODO`
- 每個 `TODO #N` 都附上**清楚的提示**：要做什麼、為什麼這樣做、有哪些常見參數
- 如果沒填 TODO 直接跑，程式會在那一行附近報錯（變數是 `None`，不會「靜默地用錯參數」）

建議先把 Step 1 跑完看到 baseline 數字後，再開始挑戰 Step 2~Step 5。

| Step | 是否需要填空 | TODO 數量 | 重點 |
|------|------------|----------|------|
| 1 | ❌ 直接執行 | 0 | 觀察 base model 的 baseline |
| 2 | ✅ 需要填空 | 3 | Teacher client、API 呼叫、抽取 response |
| 3 | ✅ 需要填空 | 2 | 抽答案 + 分類正/誤 |
| 4 | ✅ 需要填空 | 4 | **LoRA config、chat 格式化、SFTConfig、save_pretrained** |
| 5 | ✅ 需要填空 | 1 | 用 PeftModel 接回 LoRA adapter |
| 6 | ❌ 直接執行 | 0 | 對比 Before/After、填寫對比表 |

## 前置準備

你需要一組 API key（擇一）：

| Teacher | 優點 | 成本 | 合規 |
|---------|------|------|------|
| GPT-4o-mini | 品質高、穩定 | 300 題 ≈ $0.05-$0.10 | OpenAI ToS 3(c) 有限制 → 帶出討論 |
| DeepSeek（推薦）| 便宜 + MIT license | 300 題 ≈ $0.05 | MIT，無限制 |

## 操作流程

請在 `lab2/` 資料夾下，按順序執行以下步驟：

```bash
cd lab2
```

---

### Step 1：載入題庫 + 訓練前評估 — Before（30 min）🔑

```bash
uv run python step1_eval_before.py
```

**你會學到：**
- GSM8K 資料集的結構（question + 多步驟推理 + `#### 最終數字`）
- 如何從模型回答中提取最終答案（正則表達式多策略 fallback）
- Base model（1.5B）在數學推理上有多弱

**觀察重點：**
- Base model 的正確率大約多少？（預期 20-35%）
- 錯誤案例中，模型是「完全不會推理」還是「推理過程對但算錯」？

**產出檔案：** `gsm8k_before.json`

---

### Step 2：打 Teacher API 生成推理資料（40 min）🔑🔑🔑

> **這是整個課程的高潮。**
> 你第一次體驗：「用大模型的能力來訓練小模型。」

```bash
uv run python step2_generate.py
```

**⚠️ 需要完成 3 個 TODO 才能執行：**

1. `TODO #1` — 建立 `AsyncOpenAI` client + 設定 `TEACHER_MODEL`（擇方案 A 或 B、填入 API key）
2. `TODO #2` — 在 `call_api` 中呼叫 `client.chat.completions.create(...)`
3. `TODO #3` — 從 response 取出 `teacher_reasoning` 與 `usage`

**思考：** 為什麼這裡用 `AsyncOpenAI` 而不是 `OpenAI`？（提示：CONCURRENCY=20，要並行打 1000 題）

**你會學到：**
- Black-box KD 的「介面」：你只能控制 input prompt，看到 text output，無法碰 Teacher 的 logit 或 weight
- Prompt 設計對生成品質的影響
- API 成本追蹤的重要性

**思考：**
- 這個 prompt 為什麼要求 "End with: The answer is [number]"？（方便後續自動驗證）
- temperature 設 0.7 而非 0.0，你預期有什麼差異？

**產出檔案：** `synthetic_reasoning_raw.jsonl`

---

### Step 3：驗證 + 過濾 Teacher 的回答（20 min）

> Teacher 也會算錯！這一步讓你看到「大模型不是神」。

```bash
uv run python step3_verify.py
```

**⚠️ 需要完成 2 個 TODO 才能執行：**

1. `TODO #1` — 用上方寫好的 `extract_answer()` 從 `teacher_reasoning` 抽出數字
2. `TODO #2` — `if teacher_answer == gt:` 把資料分流到 `correct_data` / `wrong_data`

**你會學到：**
- 為什麼業界的 KD pipeline 一定要有驗證環節
- GPT-4o-mini 在小學數學上大約 93-95% 正確率，所以 300 題會有 15-20 題算錯
- 只用答對的資料來訓練（data quality > quantity）

**討論：**
- 如果不做這步驗證，把錯誤答案也拿去訓練 Student，會怎樣？
- 這就是為什麼 Open-R1 專案要用 Math-Verify 工具自動驗證答案

**產出檔案：** `synthetic_reasoning_verified.jsonl`、`verification_stats.json`

---

### Step 4：格式化 + 訓練 Student（50 min）⭐ 訓練流程的主菜

```bash
uv run python step4_train.py
```

**⚠️ 需要完成 4 個 TODO 才能執行（這是整個 lab 訓練流程的核心）：**

1. `TODO #1` — 建立 `LoraConfig`（r、lora_alpha、target_modules、lora_dropout）
2. `TODO #2` — 在 `format_reasoning_chat` 把資料組成 user / assistant 兩輪對話
3. `TODO #3` — 建立 `SFTConfig`（epochs、batch、grad accum、lr、max_length 等）
4. `TODO #4` — 訓練完用 `model.save_pretrained("lab2_reasoning_adapter")` 存 adapter

**你會學到：**
- 如何把 Teacher 的推理過程格式化為 SFT 訓練資料（chat template）
- CoT 資料較長，需要更大的 `max_length` 和更小的 batch size
- 推理任務用較高的 LoRA rank（r=32 vs Lab 1 的 r=16）

**關鍵設定（提示，已寫在程式碼註解內）：**
- 訓練資料：~250 筆（過濾後的高品質 CoT）
- LoRA rank：32
- Epochs：3
- max_length：4096

**如果遇到 OOM：**
1. 降 `max_length=2048`（犧牲長 CoT）
2. 降 `per_device_train_batch_size=1`
3. 降 LoRA `r=16`

**產出檔案：** `lab2_reasoning_adapter/`

---

### Step 5：訓練後評估 — After（20 min）🔑

```bash
uv run python step5_eval_after.py
```

**⚠️ 需要完成 1 個 TODO 才能執行：**

1. `TODO` — 用 `PeftModel.from_pretrained(base_model, "lab2_reasoning_adapter")` 把 step4 訓練好的 adapter 接回 base model

**陷阱：** 如果這行沒寫，你會用「base model」去評估，結果就跟 step1 的 BEFORE 一樣，看不到訓練效果。

**觀察重點：**
- 正確率從 Before 的 ~25% 提升到多少？（預期可到 50%+）
- 相比 Lab 1，只用 ~250 筆資料就能有這麼大的提升，說明了什麼？

**產出檔案：** `gsm8k_after.json`

---

### Step 6：綜合對比 + 討論（20 min）

```bash
uv run python step6_compare.py
```

**程式會產出：**
- Before vs After 正確率對比
- Student 達到 Teacher 幾成功力
- Before 錯 → After 對的題目質性分析
- 三方對比表（請填寫空白處）

**討論問題：**

**關於 Black-box KD 本身：**
1. Lab 2 的資料量小於 Lab 1（~900 vs 3,000），效果誰更明顯，為什麼？
2. Teacher 也會算錯（你看到了），如果不做 Step 3 的驗證會怎樣？
3. 生成推理資料時 temperature 設 0.7 vs 0.0，你預期結果會不同嗎？

**前沿思考：**
4. DeepSeek-R1 用了 800K 筆推理資料蒸餾出 6 個模型。我們只用了 1K 筆就看到效果了。你覺得 scaling 的甜蜜點在哪裡？
5. 我們只蒸餾了數學推理能力，如果要蒸餾寫程式的推理能力，pipeline 需要改什麼？


