"""
Lab 2 — Step 4：格式化 + 訓練 Student（50 min）
把 Teacher 生成的推理過程作為 label，訓練 Student 學會 step-by-step reasoning。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import numpy as np
import json

# ============================================
#  載入過濾後的推理資料
# ============================================

correct_data = []
with open("synthetic_reasoning_verified.jsonl", encoding="utf-8") as f:
    for line in f:
        correct_data.append(json.loads(line))

print(f"載入 {len(correct_data)} 筆已驗證的推理資料")

# ============================================
#  載入 Student + 加上 LoRA
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# TODO #1：建立 LoraConfig
# 提示：
#   - r（LoRA rank）：推理任務比 Lab 1 的 SFT 更難，建議用 r=32（Lab 1 是 16）
#   - lora_alpha：常見作法是設成跟 r 一樣（alpha/r = 1）
#   - target_modules："all-linear" 會自動把所有 Linear 層都掛上 LoRA
#   - lora_dropout=0 即可（資料量小、訓練步數少，不需要 dropout）
#   - 思考：為什麼推理任務要更高的 rank？
#     （CoT 涉及多步邏輯、變數追蹤，需要更大的可訓練參數量來承載）
lora_config = LoraConfig(
    # TODO：填入 r, lora_alpha, target_modules, lora_dropout
    

    r=32,                       # 較高的 rank 以承載複雜的推理邏輯
    lora_alpha=32,              # 常見設法 alpha = r
    target_modules="all-linear", # 覆蓋所有線性層，學習效果最完整
    lora_dropout=0,             # 小資料集通常不需 dropout
    bias="none",
    task_type="CAUSAL_LM",
    
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ============================================
#  格式化訓練資料
# ============================================


def format_reasoning_chat(item):
    # TODO #2：把一筆資料組成 chat-format 的訓練樣本
    # 提示：
    #   - SFT 訓練是讓模型「模仿 assistant 的回覆」，所以要組成兩輪對話：
    #       role="user"      → 把題目當作學生問的問題（記得加上「step by step」這類指令）
    #       role="assistant" → 用 Teacher 的完整推理過程 item["teacher_reasoning"] 當 label
    #   - 用 tokenizer.apply_chat_template(messages, tokenize=False) 把 list of dict
    #     轉成 Qwen 對應的字串格式（含 <|im_start|>、<|im_end|> 等特殊 token）
    #   - 回傳 dict 必須包含 "text" 這個 key（下面 SFTConfig.dataset_text_field="text" 會用到）
    messages = [
        # TODO：填入兩筆 message
        
        {"role": "user", "content": f"Please solve this math problem step by step:\n{item['question']}"},
        {"role": "assistant", "content": item["teacher_reasoning"]}
        
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


ds_reasoning = Dataset.from_list(correct_data).map(format_reasoning_chat)

print(f"\n=== 訓練資料範例 ===")
print(ds_reasoning[0]["text"][:500])
print("...")

token_lens = [len(tokenizer.encode(t)) for t in ds_reasoning["text"][:50]]
print(
    f"\n訓練樣本 token 長度: mean={np.mean(token_lens):.0f}, "
    f"max={np.max(token_lens)}, median={np.median(token_lens):.0f}"
)

# ============================================
#  訓練
# ============================================

# TODO #3：建立 SFTTrainer + SFTConfig
# 提示（請根據註解填入對應數值）：
#   - num_train_epochs=3                  ：跑 3 個 epoch 通常夠了
#   - save_strategy="epoch"               ：每個 epoch 存一次 checkpoint
#   - per_device_train_batch_size=2       ：CoT 資料很長，batch 設小一點避免 OOM
#   - gradient_accumulation_steps=8       ：用梯度累積補回 effective batch size = 16
#   - warmup_steps=10                     ：暖機 10 步
#   - learning_rate=2e-4                  ：LoRA 常見學習率（比 full FT 大很多）
#   - bf16/fp16                            ：依 GPU 支援度自動切換
#   - logging_steps=10                    ：每 10 步印一次 loss
#   - output_dir="outputs_lab2_reasoning" ：checkpoint 寫到這個資料夾
#   - dataset_text_field="text"           ：對應 format_reasoning_chat() 回傳的 "text" 欄位
#   - max_length=4096                     ：CoT 比一般指令長，要把 context length 拉大
#
# 若遇到 OOM，可以：
#   1) 把 max_length 降到 2048
#   2) 把 per_device_train_batch_size 降到 1
#   3) 把 LoRA 的 r 降到 16
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds_reasoning,
    args=SFTConfig(
        # TODO：填入上面註解列出的所有訓練超參
        
        num_train_epochs=3,
        save_strategy="epoch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 總 batch size = 2 * 8 = 16
        warmup_steps=10,
        learning_rate=2e-4,
        bf16=True,                      # 如果 GPU 不支援可以改為 fp16=True
        logging_steps=10,
        output_dir="outputs_lab2_reasoning",
        dataset_text_field="text",
        max_length=4096,            # 推理過程較長，需較大的 context window
        report_to="none",               # 關閉 wandb 等外部回報 (可選)
        
    ),
)

trainer.train()

# ============================================
#  儲存 adapter
# ============================================

# TODO #4：把訓練好的 LoRA adapter 存到 "lab2_reasoning_adapter" 資料夾
# 提示：
#   - 用 peft model 提供的 model.save_pretrained("路徑") 即可
#   - 只會存 adapter 權重（幾十 MB），不會把整個 base model 一起存
#   - 之後 step5 會用 PeftModel.from_pretrained(base_model, "lab2_reasoning_adapter") 載回來
# TODO：呼叫 model.save_pretrained(...)

model.save_pretrained("lab2_reasoning_adapter")

print("\n✅ 訓練完成，adapter 已儲存至 lab2_reasoning_adapter/")
