"""
Lab 2 — Step 1：載入題庫 + 訓練前評估 — Before（30 min）
載入 GSM8K 數學題庫，評估 base Qwen2.5-1.5B 的裸跑正確率。
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import random
import json
import numpy as np
from tqdm import tqdm

random.seed(42)

# ============================================
#  載入 GSM8K
# ============================================

gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_train = gsm8k["train"]  # 7,473 題 — 用來生成蒸餾資料
gsm8k_test = gsm8k["test"]  # 1,319 題 — 用來評估

print(f"GSM8K train: {len(gsm8k_train)} 題")
print(f"GSM8K test:  {len(gsm8k_test)} 題")

print("\n=== 範例 ===")
print(f"問題: {gsm8k_test[0]['question']}")
print(f"答案: {gsm8k_test[0]['answer']}")


# ============================================
#  答案提取工具函式
# ============================================


def extract_answer(text):
    """從模型回答中提取最終數字"""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].replace(",", "").strip()

    cn_match = re.search(r"答案[是为為：:]\s*([\d,]+(?:\.\d+)?)", text)
    if cn_match:
        return cn_match.group(1).replace(",", "").strip()

    en_matches = re.findall(
        r"(?:the answer is|answer is)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if en_matches:
        return en_matches[-1].replace(",", "").strip()

    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def extract_gt_answer(answer_text):
    """從 GSM8K 的 answer 欄位提取 ground truth 數字"""
    return answer_text.split("####")[-1].strip().replace(",", "")


# ============================================
#  GSM8K 評估函式
# ============================================


def evaluate_gsm8k(model, tokenizer, test_data, n_samples=50, tag=""):
    """GSM8K 正確率評估"""
    correct = 0
    total = 0
    details = []

    indices = random.sample(range(len(test_data)), min(n_samples, len(test_data)))

    for idx in tqdm(indices):
        item = test_data[idx]
        question = item["question"]
        gt = extract_gt_answer(item["answer"])

        messages = [
            {
                "role": "user",
                "content": (
                    "Solve this math problem step by step. "
                    "Show your reasoning, then give the final answer.\n\n"
                    f"{question}"
                ),
            }
        ]
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = enc["input_ids"].to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        pred = extract_answer(response)

        is_correct = pred == gt
        if is_correct:
            correct += 1
        total += 1

        details.append(
            {
                "question": question,
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "response": response,
            }
        )

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"📊 GSM8K 評估 [{tag}]")
    print(f"{'=' * 50}")
    print(f"  正確: {correct}/{total} = {accuracy:.1%}")

    wrong = [d for d in details if not d["correct"]]
    if wrong:
        print(f"\n  ❌ 錯誤範例（前 3 個）:")
        for w in wrong[:3]:
            print(f"    Q: {w['question'][:80]}...")
            print(f"    預測: {w['pred']}, 正確: {w['gt']}")
            print()

    return accuracy, details


# ============================================
#  Before 評估
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print("🔍 Before 評估：base Qwen2.5-1.5B 在 GSM8K 上的表現")
before_acc, before_details = evaluate_gsm8k(
    model, tokenizer, gsm8k_test, n_samples=50, tag="BEFORE"
)

with open("gsm8k_before.json", "w") as f:
    json.dump(
        {"accuracy": before_acc, "details": before_details},
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"\n已儲存 gsm8k_before.json（預期 base model 約 20-35% 正確率）")
