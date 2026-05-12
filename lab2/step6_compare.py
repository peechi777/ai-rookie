"""
Lab 2 — Step 6：綜合對比 + 質性分析 + 討論（20 min）
Before vs After 大對比，觀察 Student 學會了什麼推理能力。
"""

import json

# ============================================
#  載入 Before / After 結果
# ============================================

with open("gsm8k_before.json") as f:
    before_data = json.load(f)

with open("gsm8k_after.json") as f:
    after_data = json.load(f)

with open("verification_stats.json") as f:
    verify_stats = json.load(f)

before_acc = before_data["accuracy"]
after_acc = after_data["accuracy"]
before_details = before_data["details"]
after_details = after_data["details"]
teacher_accuracy = verify_stats["teacher_accuracy"]

# ============================================
#  Before vs After 大對比
# ============================================

print("\n" + "=" * 60)
print("🏆 GSM8K 正確率 Before vs After")
print("=" * 60)
print(f"  Before (base model):  {before_acc:.1%}")
print(f"  After  (distilled):   {after_acc:.1%}")
print(f"  提升:                 {after_acc - before_acc:+.1%}")
relative_gain = (after_acc - before_acc) / max(before_acc, 0.01) * 100
print(f"  相對提升:             {relative_gain:+.0f}%")
print()
print(f"  Teacher 正確率:       {teacher_accuracy:.1%}")
teacher_ratio = after_acc / max(teacher_accuracy, 0.01) * 100
print(f"  Student 達到 Teacher 的: {teacher_ratio:.0f}%")

# ============================================
#  質性分析：Before 錯 → After 對
# ============================================

before_wrong = {d["question"]: d for d in before_details if not d["correct"]}
after_right = {d["question"]: d for d in after_details if d["correct"]}

improved = []
for q in before_wrong:
    if q in after_right:
        improved.append((before_wrong[q], after_right[q]))

print(f"\n🎯 Before 錯 → After 對 的題目: {len(improved)} 個")
for b, a in improved[:3]:
    print(f"\n  Q: {b['question'][:80]}...")
    print(f"  Before 回答: {b['response'][:150]}...")
    print(f"  After  回答: {a['response'][:150]}...")
    print(f"  正確答案: {b['gt']}")

# ============================================
#  學生填寫的三方對比表
# ============================================

print("\n" + "=" * 60)
print("📋 三方對比表（請填寫空白處）")
print("=" * 60)
print(f"""
| 維度               | Lab 1 (SFT)         | Lab 2 (Reasoning KD)    |
|--------------------|---------------------|-------------------------|
| 資料來源           | 人類標註            | Teacher API 生成         |
| 資料包含推理過程？ | ❌ 只有最終回答      | ✅ 包含 step-by-step     |
| 評估指標           | ROUGE-L + 人工打分  | GSM8K 正確率（客觀）     |
| Before 分數        | ____                | {before_acc:.1%}         |
| After 分數         | ____                | {after_acc:.1%}          |
| 訓練資料量         | 3,000               | {verify_stats['correct']}  |
| 訓練成本（API）    | $0 (用現成資料集)   | $____                    |
| 訓練時間           | ~__ min             | ~__ min                  |
""")

