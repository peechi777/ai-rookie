"""
Lab 2 — Step 5：訓練後評估 — After（20 min）
用完全相同的 seed、相同的 50 題評估蒸餾後的 Student。
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re
import random
import json
from tqdm import tqdm

random.seed(42)

# ============================================
#  載入 GSM8K 測試集
# ============================================

gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_test = gsm8k["test"]


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
    return answer_text.split("####")[-1].strip().replace(",", "")


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
#  載入訓練後的模型
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# TODO：把 step4 訓練好的 LoRA adapter 套到 base model 上
# 提示：
#   - 我們在 step4 用 model.save_pretrained("lab2_reasoning_adapter") 存過 adapter
#   - 這裡要先載入 base model（上面已做好），再用 PeftModel.from_pretrained 把 adapter 接上去
#   - 寫法：model = PeftModel.from_pretrained(base_model, "adapter 資料夾路徑")
#   - 思考：如果這一步直接拿掉，模型就會是 base 版本，跟 step1 的 BEFORE 結果一樣
# TODO：用 PeftModel.from_pretrained(...) 重新賦值給 model

model = PeftModel.from_pretrained(model, "lab2_reasoning_adapter")

model.eval()

# ============================================
#  After 評估
# ============================================

print("🔍 After 評估：蒸餾後的 Student 在 GSM8K 上的表現")
after_acc, after_details = evaluate_gsm8k(
    model, tokenizer, gsm8k_test, n_samples=50, tag="AFTER"
)

with open("gsm8k_after.json", "w") as f:
    json.dump(
        {"accuracy": after_acc, "details": after_details},
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"\n已儲存 gsm8k_after.json")
