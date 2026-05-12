"""
Lab 1 — Step 4：SFT 訓練（40 min）
用 QLoRA + TRL 對 Qwen2.5-1.5B 做 SFT 微調。
訓練資料取 3,000 筆子集（16GB VRAM 限制）。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import json

# ============================================
#  載入資料集 + 切割 indices
# ============================================

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)

with open("split_indices.json") as f:
    split_info = json.load(f)

ds_train = ds.select(split_info["train_indices"])
print(f"訓練集大小: {len(ds_train)}")

# ============================================
#  載入模型 + 加上 LoRA adapter
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# TODO #1：建立 LoraConfig（Lab 1 標準設定）
# 提示：
#   - r=16（rank）：Lab 1 是一般指令跟隨任務，rank 不需要太高
#     （Lab 2 推理任務才會用 r=32）
#   - lora_alpha=16：常見作法是設成跟 r 一樣
#   - target_modules：Qwen 架構需要掛 LoRA 的 Linear 層名稱
#       attention：q_proj, k_proj, v_proj, o_proj
#       MLP：     gate_proj, up_proj, down_proj
#     （也可以寫 "all-linear" 讓 peft 自動偵測；這裡用顯式列表展示哪些層在被訓練）
#   - lora_dropout=0：資料量夠（6K 筆 × 3 epoch），不需要 dropout
#   - 思考：為什麼要把 LoRA 掛在 attention + MLP 兩種層上？只掛 attention 行不行？
#     （行，但 MLP 是 Transformer 中參數最多的部分，不掛會錯過很多容量）
lora_config = LoraConfig(
    # TODO：填入 r, lora_alpha, target_modules, lora_dropout
    
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    task_type="CAUSAL_LM",
    
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ============================================
#  格式化訓練資料
# ============================================

SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "Be concise, professional, and empathetic."
)


def format_chat(example):
    # TODO #2：把一筆資料組成「三輪對話」當作 SFT 訓練樣本
    # 提示：
    #   - SFT 要教模型「給定 system + user，學會 assistant 該怎麼回」，所以是三段：
    #       system   → SYSTEM_PROMPT（角色設定）
    #       user     → example["instruction"]（客戶問題）
    #       assistant → example["response"]（標準答案，模型要學的目標）
    #   - 用 tokenizer.apply_chat_template(messages, tokenize=False) 把 list of dict 轉成字串
    #     （tokenize=False 表示「先不 tokenize」，等 SFTTrainer 自己處理）
    #   - 必須回傳 {"text": ...}，因為下面 SFTConfig 設定了 dataset_text_field="text"
    #   - ⚠️ system prompt 必須跟 step3/step5 評估時用的「開頭一致」，否則訓練/評估條件不同
    messages = [
        # TODO：填入 system / user / assistant 三筆 message
        
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]}
        
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds_sft = ds_train.shuffle(seed=42).select(range(6000))
ds_sft = ds_sft.map(format_chat)

print(f"訓練子集大小: {len(ds_sft)}")
print(f"\n=== 訓練資料範例 ===")
print(ds_sft[0]["text"][:500])

# ============================================
#  訓練
# ============================================

# TODO #3：建立 SFTTrainer + SFTConfig
# 提示（請根據註解填入對應數值）：
#   - num_train_epochs=3                  ：跑 3 個 epoch（會被 max_steps 限制提前停）
#   - per_device_train_batch_size=4       ：客服回答比 CoT 短，batch 可以比 Lab 2 大
#   - gradient_accumulation_steps=4       ：effective batch size = 4 × 4 = 16
#   - warmup_steps=10                     ：暖機 10 步
#   - save_strategy="epoch"               ：每 epoch 存一次 checkpoint
#   - max_steps=600                       ：時間有限，硬性限制 600 步避免訓練太久
#   - learning_rate=2e-4                  ：LoRA 常見學習率
#   - bf16/fp16                            ：依 GPU 支援度自動切換
#       bf16=torch.cuda.is_bf16_supported(),
#       fp16=not torch.cuda.is_bf16_supported(),
#   - logging_steps=10                    ：每 10 步印一次 loss
#   - output_dir="outputs_lab1"           ：checkpoint 寫到這
#   - dataset_text_field="text"           ：對應 format_chat() 回傳的 "text" 欄位
#   - max_length=2048                     ：客服回答不長，2048 夠
#
# 若遇到 OOM，可以：
#   1) 把 per_device_train_batch_size 降到 2
#   2) 把 LoRA 的 r 降到 8
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds_sft,
    args=SFTConfig(
        # TODO：填入上面註解列出的所有訓練超參
        
        output_dir="outputs_lab1",
        num_train_epochs=3,
        max_steps=600,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
        max_length=2048,
    ),
)

trainer.train()

# ============================================
#  儲存 adapter
# ============================================

# TODO #4：把訓練好的 LoRA adapter 存到 "lab1_customer_support_adapter" 資料夾
# 提示：
#   - 直接呼叫 model.save_pretrained("路徑")
#   - 只會存 adapter 權重（幾十 MB），不會把 base model 一起存
#   - 之後 step5 會用 PeftModel.from_pretrained(base_model, "lab1_customer_support_adapter") 載回
# TODO：呼叫 model.save_pretrained(...)

model.save_pretrained("lab1_customer_support_adapter")

print("\n✅ 訓練完成，adapter 已儲存至 lab1_customer_support_adapter/")
