"""
Lab 1 — Step 1：環境 + 資料探索（20 min）
載入 Bitext 客服資料集，觀察欄位結構、意圖分布、回答長度分布。
"""

from datasets import load_dataset
import numpy as np
from collections import Counter

# ============================================
#  載入資料集
# ============================================

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)
print(f"總資料量: {len(ds)}")
print(f"欄位: {ds.column_names}")
print()

# ============================================
#  看一筆資料
# ============================================

print("=== 範例 ===")
print(f"類別: {ds[0]['category']}")
print(f"意圖: {ds[0]['intent']}")
print(f"問題: {ds[0]['instruction']}")
print(f"回答: {ds[0]['response']}")
print()

# ============================================
#  意圖分布
# ============================================

intent_counts = Counter(ds["intent"])
print(f"意圖類別數: {len(intent_counts)}")
for intent, count in intent_counts.most_common(5):
    print(f"  {intent}: {count}")

# ============================================
#  回答長度分布
# ============================================

resp_lens = [len(r.split()) for r in ds["response"]]
print(
    f"\n回答長度: mean={np.mean(resp_lens):.0f}, "
    f"median={np.median(resp_lens):.0f}, "
    f"max={np.max(resp_lens)}"
)
