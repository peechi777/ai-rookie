"""
Lab 1 — Step 2：切出評估集（10 min）
按意圖做分層抽樣，確保每個意圖在測試集中都有樣本。
評估集在訓練之前就切好、鎖死，後續不能碰。
"""

from datasets import load_dataset
from collections import Counter, defaultdict
import random
import json

random.seed(42)

# ============================================
#  載入資料集
# ============================================

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)

# ============================================
#  分層抽樣：每個 intent 取 10% 做 test
# ============================================

# TODO #1：建立 intent → indices 的對應表
# 提示：
#   - intent_groups 是個 dict，key 是 intent 名稱，value 是該 intent 的所有資料 index
#   - 用 enumerate(ds) 走過資料集，把每筆 item["intent"] 對應的 index 累積到對應的 list
#   - defaultdict(list) 可以省掉「key 不存在時要先建立 list」的判斷
#   - 思考：為什麼不直接 random split？
#     （因為有些罕見 intent 樣本只有幾十筆，隨機切會有 intent 完全沒出現在 test）
intent_groups = defaultdict(list)
# TODO：把每一筆資料的 index 累積到對應 intent 的 list

for idx, item in enumerate(ds):
    intent = item["intent"]
    intent_groups[intent].append(idx)

train_indices = []
test_indices = []

# TODO #2：對「每一個 intent」獨立做 90/10 切分（分層抽樣）
# 提示：
#   - 對 intent_groups.items() 跑迴圈
#   - 每個 intent 內先 random.shuffle(indices) 打散順序
#   - n_test = max(5, int(len(indices) * 0.1))  ← 至少留 5 筆做 test，避免太小
#   - 前 n_test 筆 → test_indices；剩下的 → train_indices
#   - 用 .extend() 而不是 .append()（後者會把整個 list 當成一個元素）
# TODO：for intent, indices in intent_groups.items(): ...

for intent, indices in intent_groups.items():
    # 1. 確保每個意圖內部的順序是隨機的
    random.shuffle(indices)
    
    # 2. 計算測試集需要的數量 (至少 5 筆)
    n_test = max(5, int(len(indices) * 0.1))
    
    # 3. 切割
    test_slice = indices[:n_test]
    train_slice = indices[n_test:]
    
    # 4. 加入總表
    test_indices.extend(test_slice)
    train_indices.extend(train_slice)

ds_train = ds.select(train_indices)
ds_test = ds.select(test_indices)

print(f"訓練集: {len(ds_train)}")
print(f"測試集: {len(ds_test)}")

# ============================================
#  驗證分層結果
# ============================================

test_intents = Counter(ds_test["intent"])
print(f"測試集涵蓋 {len(test_intents)} 個意圖")

# ============================================
#  儲存切割好的 indices（供後續 step 使用）
# ============================================

split_info = {
    "train_indices": train_indices,
    "test_indices": test_indices,
    "train_size": len(ds_train),
    "test_size": len(ds_test),
    "n_intents_in_test": len(test_intents),
}
with open("split_indices.json", "w") as f:
    json.dump(split_info, f, indent=2)

print("\n已儲存 split_indices.json，供後續 step 載入使用。")
