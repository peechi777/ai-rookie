"""
Lab 1 — Step 6：Before vs After 對比報告 + 討論（5 min）
載入 before/after 評估結果，產生完整對比報告。
"""

from collections import defaultdict
import numpy as np
import json

# ============================================
#  載入 Before / After 評估結果
# ============================================

with open("eval_before.json") as f:
    before_results = json.load(f)

with open("eval_after.json") as f:
    after_results = json.load(f)

# ============================================
#  整體對比
# ============================================

before_rouge = np.mean([r["rouge_l"] for r in before_results])
after_rouge = np.mean([r["rouge_l"] for r in after_results])
before_len = np.mean([r["len_ratio"] for r in before_results])
after_len = np.mean([r["len_ratio"] for r in after_results])

print("\n" + "=" * 60)
print("📊 BEFORE vs AFTER 對比")
print("=" * 60)
print(f"{'指標':<15} {'Before':>10} {'After':>10} {'變化':>10}")
print(f"{'-' * 45}")
print(
    f"{'ROUGE-L':<15} {before_rouge:>10.4f} {after_rouge:>10.4f} "
    f"{after_rouge - before_rouge:>+10.4f}"
)
print(
    f"{'長度比':<15} {before_len:>10.2f} {after_len:>10.2f} "
    f"{after_len - before_len:>+10.2f}"
)

# ============================================
#  逐意圖對比
# ============================================

before_by_intent = defaultdict(list)
after_by_intent = defaultdict(list)
for r in before_results:
    before_by_intent[r["intent"]].append(r["rouge_l"])
for r in after_results:
    after_by_intent[r["intent"]].append(r["rouge_l"])

print(f"\n各意圖進步幅度:")
deltas = {}
for intent in before_by_intent:
    if intent in after_by_intent:
        b = np.mean(before_by_intent[intent])
        a = np.mean(after_by_intent[intent])
        deltas[intent] = a - b

sorted_deltas = sorted(deltas.items(), key=lambda x: x[1])
print("  進步最少:")
for intent, d in sorted_deltas[:3]:
    print(f"    {intent}: {d:+.4f}")
print("  進步最多:")
for intent, d in sorted_deltas[-3:]:
    print(f"    {intent}: {d:+.4f}")


