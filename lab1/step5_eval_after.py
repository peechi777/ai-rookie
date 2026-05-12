"""
Lab 1 — Step 5：訓練後評估 — After（20 min）
載入訓練後的模型，用完全相同的函式 + 完全相同的測試集進行評估。
同時對 step3 人工打分的 10 題重新生成 after 版本回答。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from peft import PeftModel
from collections import defaultdict
import torch
import numpy as np
import random
import json

random.seed(42)

# ============================================
#  載入資料集 + 切割 indices
# ============================================

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)

with open("split_indices.json") as f:
    split_info = json.load(f)

ds_test = ds.select(split_info["test_indices"])
print(f"測試集大小: {len(ds_test)}")

# ============================================
#  載入訓練後的模型
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "lab1_customer_support_adapter"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# TODO：把 step4 訓練好的 LoRA adapter 套到 base model 上
# 提示：
#   - 我們在 step4 用 model.save_pretrained("lab1_customer_support_adapter") 存過 adapter
#   - 這裡要把 adapter 接回 base model：
#       model = PeftModel.from_pretrained(base_model, "adapter 資料夾路徑")
#   - 思考：如果這行直接拿掉，下面的 AFTER 評估會跟 step3 的 BEFORE 一樣，看不出訓練效果
#   - 注意：下面「人工打分」段落也是用同一個 model 物件，所以這行非寫不可
# TODO：用 PeftModel.from_pretrained(...) 重新賦值給 model

model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# ============================================
#  自動評估函式（與 step3 完全相同）
# ============================================

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def evaluate_model(model, tokenizer, test_data, n_samples=100, tag=""):
    """對模型進行完整評估"""
    results = []
    indices = random.sample(range(len(test_data)), min(n_samples, len(test_data)))

    for idx in indices:
        item = test_data[idx]
        question = item["instruction"]
        reference = item["response"]
        intent = item["intent"]

        messages = [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": question},
        ]
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor) else enc).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        prediction = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        rouge = scorer.score(reference, prediction)
        rouge_l = rouge["rougeL"].fmeasure
        len_ratio = len(prediction.split()) / max(len(reference.split()), 1)

        results.append(
            {
                "intent": intent,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "rouge_l": rouge_l,
                "len_ratio": len_ratio,
            }
        )

    avg_rouge = np.mean([r["rouge_l"] for r in results])
    avg_len_ratio = np.mean([r["len_ratio"] for r in results])

    print(f"\n{'=' * 50}")
    print(f"📊 評估結果 [{tag}]")
    print(f"{'=' * 50}")
    print(f"  樣本數:      {len(results)}")
    print(f"  ROUGE-L:     {avg_rouge:.4f}")
    print(f"  長度比:      {avg_len_ratio:.2f}x (1.0 = 與 reference 等長)")

    intent_scores = defaultdict(list)
    for r in results:
        intent_scores[r["intent"]].append(r["rouge_l"])

    print(f"\n  各意圖 ROUGE-L（前 5 低 / 前 5 高）:")
    sorted_intents = sorted(intent_scores.items(), key=lambda x: np.mean(x[1]))
    for intent, scores in sorted_intents[:5]:
        print(f"    ❌ {intent}: {np.mean(scores):.4f}")
    print("    ...")
    for intent, scores in sorted_intents[-5:]:
        print(f"    ✅ {intent}: {np.mean(scores):.4f}")

    return results


# ============================================
#  執行 After 評估
# ============================================

print("🔍 開始 AFTER 評估（訓練後）...")
after_results = evaluate_model(
    model, tokenizer, ds_test, n_samples=100, tag="AFTER"
)

with open("eval_after.json", "w") as f:
    json.dump(after_results, f, ensure_ascii=False, indent=2)

print("\n已儲存 eval_after.json")

# ============================================
#  人工打分（相同 10 題的 after 版本）
# ============================================

with open("manual_eval_questions.json") as f:
    manual_questions = json.load(f)

with open("eval_before.json") as f:
    before_results_all = json.load(f)

before_by_question = {r["question"]: r for r in before_results_all}

print("\n" + "=" * 60)
print("📝 人工評估（AFTER）：相同的 10 題")
print("=" * 60)

for i, mq in enumerate(manual_questions):
    messages = [
        {"role": "system", "content": "You are a helpful customer support agent."},
        {"role": "user", "content": mq["question"]},
    ]
    enc = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor) else enc).to("cuda")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
    after_pred = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=True
    )

    before_pred = before_by_question.get(mq["question"], {}).get("prediction", "N/A")
    before_ref = before_by_question.get(mq["question"], {}).get("reference", "N/A")

    print(f"\n--- 題 {i + 1} [{mq['intent']}] ---")
    print(f"🙋 客戶:     {mq['question']}")
    print(f"🤖 Before:   {before_pred[:150]}...")
    print(f"🤖 After:    {after_pred[:150]}...")
    print(f"📖 Reference: {before_ref[:150]}...")
    print("   👉 Before 分數: ___  After 分數: ___")
