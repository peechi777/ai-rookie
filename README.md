# LLM 知識蒸餾實作 Lab

> **硬體需求：** NVIDIA GPU（建議 RTX 4070 Ti 16GB VRAM 以上）  
> **總時長：** 約 5 小時  
> **工具鏈：** Transformers + PEFT (LoRA) + TRL + OpenAI SDK  
> **Student 模型：** Qwen2.5-1.5B

## 專案總覽

本專案包含兩個循序漸進的 Lab，帶你從 SFT 基礎到完整的 Black-box 知識蒸餾 pipeline。

| | Lab 1：SFT Baseline | Lab 2：Black-box Reasoning KD |
|---|---|---|
| **資料來源** | 現成資料集（人工標註） | 自己打 Teacher API 合成 |
| **訓練目標** | 學會特定領域的回答風格 | 學會推理過程（CoT） |
| **評估方式** | Intent 正確率 + 回答品質打分 | GSM8K 數學正確率（客觀可驗證） |
| **教學目的** | 理解 SFT 流程 + 建立 baseline 感覺 | 體驗完整 KD pipeline + 感受 CoT 蒸餾的威力 |

## 環境安裝

```bash
# 安裝 uv（如果尚未安裝）
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS
# curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步環境
uv sync
```

## 專案結構

```
.
├── pyproject.toml          # 依賴管理（uv sync）
├── README.md               # 本文件
├── lab1/                   # Lab 1: SFT 客服模型
│   ├── README.md           # Lab 1 教學引導
│   ├── step1_explore.py    # 環境 + 資料探索
│   ├── step2_split.py      # 切出評估集
│   ├── step3_eval_before.py# 訓練前評估
│   ├── step4_train.py      # SFT 訓練
│   ├── step5_eval_after.py # 訓練後評估
│   └── step6_compare.py    # Before vs After 對比
└── lab2/                   # Lab 2: Reasoning KD
    ├── README.md           # Lab 2 教學引導
    ├── step1_eval_before.py# 載入題庫 + 訓練前評估
    ├── step2_generate.py   # 打 Teacher API 生成推理資料
    ├── step3_verify.py     # 驗證 + 過濾 Teacher 回答
    ├── step4_train.py      # 格式化 + 訓練 Student
    ├── step5_eval_after.py # 訓練後評估
    └── step6_compare.py    # 綜合對比 + 討論
```

## 操作方式

每個 Lab 的 `README.md` 會引導你按照 Step 1 → Step 6 依序執行。
請進入各 Lab 資料夾後，按順序執行各 step 的 `.py` 檔案：

```bash
cd lab1
uv run python step1_explore.py
uv run python step2_split.py
# ...依此類推
```

## OOM 急救

| 症狀 | 解法 |
|------|------|
| Lab 2 訓練 OOM | 降 `max_seq_length=2048`，犧牲長 CoT |
| 還是 OOM | 降 `per_device_train_batch_size=1` |
| 還是 OOM | 降 LoRA `r=16` |
| API 太慢 | 減少生成量到 100 題，或切成小組分工 |
| Teacher 正確率太低 | 換更好的 Teacher（GPT-4o），或加 few-shot prompt |
