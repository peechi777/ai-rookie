"""
Lab 2 — Step 3：驗證 + 過濾 Teacher 的回答（20 min）
Teacher 也會算錯！這一步讓學生看到「大模型不是神」，
以及為什麼業界的 KD pipeline 一定要有驗證環節。
"""

import re
import json

# ============================================
#  載入 Teacher 生成的原始資料
# ============================================

synthetic_reasoning_data = []
with open("synthetic_reasoning_raw.jsonl", encoding="utf-8") as f:
    for line in f:
        synthetic_reasoning_data.append(json.loads(line))

print(f"載入 {len(synthetic_reasoning_data)} 筆 Teacher 生成資料")


# ============================================
#  答案提取函式
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

    # Use findall to get the last "The answer is X" match (avoids early = matches in LaTeX)
    en_matches = re.findall(
        r"(?:the answer is|answer is)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if en_matches:
        return en_matches[-1].replace(",", "").strip()

    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


# ============================================
#  驗證 Teacher 答案
# ============================================

correct_data = []
wrong_data = []

for item in synthetic_reasoning_data:
    # TODO #1：從 Teacher 的推理文字中抽出最終數字答案
    # 提示：上面已經寫好 extract_answer() 工具函式，直接呼叫即可。
    #       要解析的欄位是 item["teacher_reasoning"]。
    teacher_answer = extract_answer(item["teacher_reasoning"])  # ← 換成 extract_answer(...)

    gt = item["gt_answer"]

    # TODO #2：把 Teacher 答對的歸到 correct_data、答錯的歸到 wrong_data
    # 提示：
    #   - 答案是字串比對（注意 extract_answer 已經去掉逗號、空白）
    #   - 對的用 correct_data.append(item)、錯的用 wrong_data.append(item)
    #   - 思考：為什麼要把錯的「也存起來」而不是直接丟掉？
    #     （等等下方的列印會顯示「Teacher 算錯的範例」給你看，這就是用途）
    if teacher_answer == gt:
        correct_data.append(item)
    else:
        wrong_data.append(item)

teacher_accuracy = len(correct_data) / len(synthetic_reasoning_data)

print(
    f"\nTeacher 回答正確率: {len(correct_data)}/{len(synthetic_reasoning_data)} "
    f"= {teacher_accuracy:.1%}"
)
print(f"  ✅ 正確: {len(correct_data)}")
print(f"  ❌ 錯誤: {len(wrong_data)}")

if wrong_data:
    print(f"\n⚠️ Teacher 算錯的範例（前 3 個）:")
    for w in wrong_data[:3]:
        teacher_ans = extract_answer(w["teacher_reasoning"])
        print(f"  Q: {w['question'][:80]}...")
        print(f"  Teacher 答: {teacher_ans}, 正確答案: {w['gt_answer']}")
        print()

print(f"用於訓練的高品質資料: {len(correct_data)} 筆")

# ============================================
#  儲存過濾後的資料 + 統計
# ============================================

with open("synthetic_reasoning_verified.jsonl", "w", encoding="utf-8") as f:
    for item in correct_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

stats = {
    "total_generated": len(synthetic_reasoning_data),
    "correct": len(correct_data),
    "wrong": len(wrong_data),
    "teacher_accuracy": teacher_accuracy,
}
with open("verification_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n已儲存 synthetic_reasoning_verified.jsonl ({len(correct_data)} 筆)")
print("已儲存 verification_stats.json")

# ============================================
#  討論
# ============================================

print(f"""
💡 討論：
- GPT-4o-mini 在小學數學上大約 93-95% 正確率，所以 300 題會有 15-20 題算錯
- 如果不做這步驗證，把錯誤答案也拿去訓練 Student，會怎樣？
- 這就是為什麼 Open-R1 專案要用 Math-Verify 工具自動驗證答案
""")
