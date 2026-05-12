"""
Lab 1 — Step 3：訓練前評估 — Before（25 min）
這是整個 Lab 最重要的環節之一。沒有 before 數據，after 的數字毫無意義。

評估指標：
  - ROUGE-L：回答與 ground truth 的文字重疊度（自動、客觀）
  - 回答長度比：模型是否廢話太多或太簡短（len(pred) / len(ref)）
  - 人工品質打分：學生互評 1-5 分（抽 10 題）
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
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
#  載入 base Student（訓練前）
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ============================================
#  自動評估函式
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

        # TODO #1：把問題包成 chat 格式並交給模型生成回答
        # 提示：
        #   - 用 system + user 兩輪對話：
        #       system: "You are a helpful customer support agent."
        #       user:   question（目前的客戶問題）
        #     ⚠️ 重要：這個 system prompt 必須跟 step4_train.py 的 SYSTEM_PROMPT「開頭一致」
        #             ，否則 base model / 訓練後模型評估的條件不同，會比錯
        #   - tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        #     會回傳 dict 或 tensor（不同版本不同）；下面用「相容寫法」處理
        #   - model.generate(input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
        #   - 最後用 tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True) 解碼，
        #     要 slice 掉 input 部分，只留模型新生成的內容
        messages = [
            # TODO：填入 system + user 兩筆 message
            
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": question}
            
        ]
        enc = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = (enc["input_ids"] if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor) else enc).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                # TODO：傳入 input_ids 與 generate 參數
                
                input_ids=input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
                
            )
        prediction = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        # TODO #2：計算這一題的 ROUGE-L 分數
        # 提示：
        #   - 上方已經建好 scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        #   - 呼叫 scorer.score(reference, prediction) 會回傳 dict
        #     例如 {"rougeL": Score(precision=..., recall=..., fmeasure=...)}
        #   - 我們要的是 rouge["rougeL"].fmeasure（F1 分數，0~1 之間）
        #   - 思考：為什麼不只看 precision？
        #     （precision 高代表「沒講廢話」，但「沒回答」也會 precision 高，所以要看 F1）
        rouge = scorer.score(reference, prediction)       # ← 換成 scorer.score(...)
        rouge_l = rouge["rougeL"].fmeasure     # ← 從 rouge 中取 rougeL.fmeasure
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
#  執行 Before 評估
# ============================================

print("🔍 開始 BEFORE 評估（訓練前的 base model）...")
before_results = evaluate_model(
    model, tokenizer, ds_test, n_samples=100, tag="BEFORE"
)

with open("eval_before.json", "w") as f:
    json.dump(before_results, f, ensure_ascii=False, indent=2)

print("\n已儲存 eval_before.json")

# ============================================
#  人工打分（抽 10 題）
# ============================================

print("\n" + "=" * 60)
print("📝 人工評估：請為以下回答打分（1-5 分）")
print("   1=完全無用  2=答非所問  3=部分有用  4=大致正確  5=專業完整")
print("=" * 60)

sample_10 = random.sample(before_results, 10)

# 儲存抽中的 10 題（供 step5 對比使用）
with open("manual_eval_questions.json", "w") as f:
    json.dump(
        [{"intent": r["intent"], "question": r["question"]} for r in sample_10],
        f,
        ensure_ascii=False,
        indent=2,
    )

for i, r in enumerate(sample_10):
    print(f"\n--- 題 {i + 1} [{r['intent']}] ---")
    print(f"🙋 客戶: {r['question']}")
    print(f"🤖 模型: {r['prediction'][:200]}...")
    print(f"📖 參考: {r['reference'][:200]}...")
    print(f"   ROUGE-L: {r['rouge_l']:.4f}")
    print("   👉 你的打分（1-5）: ___")
