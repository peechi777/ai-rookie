import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# 關鍵：解決顯存碎片化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "/home/user/model/Llama-3.2-1B-Instruct"

print("📥 正在以 bfloat16 模式載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 4070 必用，省下一半顯存空間
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = [
    AWQModifier(
        targets=["Linear"],
        scheme="W4A16",
        ignore=["lm_head"]
    )
]

print("🚀 開始執行 512 樣本 GPU AWQ 校準...")
oneshot(
    model=model,
    recipe=recipe,
    dataset="open_platypus",
    num_calibration_samples=512, # 這次挑戰 512！
    max_seq_length=512,          # 長度設 512 以保險
    output_dir="./Llama-3.2-1B-Instruct-AWQ-GPU-P512"
)
print("✅ GPU 高精度量化完成！")