"""
Lab 2 — Step 2：打 Teacher API 生成推理資料（40 min）🔑🔑🔑
這是整個課程的高潮。
學生第一次體驗：「我在用大模型的能力來訓練小模型。」

Teacher 選擇（擇一設定）：
  方案 A — GPT-4o-mini：品質高、穩定，500 題 ≈ $0.10
  方案 B — DeepSeek（推薦）：便宜 + MIT license，500 題 ≈ $0.05
"""

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
import asyncio
import random
import json

random.seed(42)

# ============================================
#  選擇 Teacher（TODO #1：擇一設定，並填入 API key）
# ============================================
#
# 提示：
#   - 兩個方案都用 OpenAI 相容的 API 介面（chat.completions），差別只在 base_url 與 model 名稱
#   - 我們要做「並行 API 呼叫」，所以一定要用 AsyncOpenAI 而不是同步的 OpenAI
#   - DeepSeek 需要額外傳 base_url="https://api.deepseek.com"
#
# 方案 A — GPT-4o-mini：
#   client = AsyncOpenAI(api_key=...)
#   TEACHER_MODEL = "gpt-4o-mini"
#
# 方案 B — DeepSeek（推薦）：
#   client = AsyncOpenAI(api_key=..., base_url="https://api.deepseek.com")
#   TEACHER_MODEL = "deepseek-chat"

# TODO #1：建立 client、設定 TEACHER_MODEL
client = AsyncOpenAI(api_key="API") # ← 換成 AsyncOpenAI(...)

TEACHER_MODEL = "gpt-4o-mini"  # ← 換成 "gpt-4o-mini" 或 "deepseek-chat"

CONCURRENCY = 20  # 同時發出的 API requests 數量

# ============================================
#  Prompt 設計（強化版）
# ============================================

REASONING_PROMPT = """You are an expert math tutor. Solve the following elementary school math problem with a clear, detailed reasoning chain that a student can follow and learn from.

Instructions:
1. Identify all quantities and define variables explicitly at the start
2. Break the solution into numbered steps — one logical operation per step
3. For EVERY arithmetic operation, write the full equation with numbers:
   e.g. "16 - 3 - 4 = 9" or "9 × $2 = $18"
4. After each calculation, state in plain English what the result represents
5. End with a one-sentence summary of the solution path
6. Final line must be exactly: "The answer is [number]"
   (integer or decimal, no units, no commas, no extra text)

Problem:
{question}"""

# ============================================
#  載入 GSM8K 題目
# ============================================

gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_train = gsm8k["train"]


def extract_gt_answer(answer_text):
    return answer_text.split("####")[-1].strip().replace(",", "")


seed_questions = gsm8k_train.shuffle(seed=42).select(range(1000))

# ============================================
#  非同步並行生成
# ============================================


async def call_api(semaphore: asyncio.Semaphore, item: dict, idx: int):
    question = item["question"]
    gt_answer = extract_gt_answer(item["answer"])

    async with semaphore:
        try:
            # TODO #2：呼叫 Teacher API 取得推理過程
            # 提示：
            #   - 使用 await client.chat.completions.create(...)
            #   - model 用 TEACHER_MODEL
            #   - messages 是一個 list，包含一筆 {"role": "user", "content": ...}
            #     content 用 REASONING_PROMPT.format(question=question)
            #   - temperature=0.7（思考：為什麼不用 0.0？這牽涉到 Lab 後面的討論題）
            #   - max_tokens=2048（推理可能很長，要給足）
            response = None  # ← 換成 await client.chat.completions.create(...)
            
            response = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": REASONING_PROMPT.format(question=question)}],
                temperature=0.7,
                max_tokens=2048,
            )

            # TODO #3：從 API response 取出 Teacher 的推理文字 + token 用量
            # 提示：
            #   - OpenAI 相容介面的回傳格式：
            #       response.choices[0].message.content   ← 模型生成的文字
            #       response.usage.prompt_tokens          ← input token 數
            #       response.usage.completion_tokens      ← output token 數
            #   - 文字記得 .strip() 去頭尾空白
            teacher_reasoning = response.choices[0].message.content.strip()  # ← 從 response 取出
            usage = response.usage              # ← 從 response 取出

            return {
                "question": question,
                "gt_answer": gt_answer,
                "teacher_reasoning": teacher_reasoning,
                "teacher_model": TEACHER_MODEL,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            print(f"\n  ⚠️ 第 {idx} 題失敗: {e}")
            return None


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        call_api(semaphore, item, i) for i, item in enumerate(seed_questions)
    ]

    print(f"Teacher: {TEACHER_MODEL}")
    print(f"並行度: {CONCURRENCY}")
    print(f"開始生成推理資料：{len(tasks)} 題")
    print()

    # gather 全部，tqdm 顯示進度
    raw_results = await atqdm.gather(*tasks, desc="打 Teacher API")

    results = [r for r in raw_results if r is not None]
    errors = len(tasks) - len(results)
    print(f"\n✅ 生成完成: {len(results)} 筆成功, {errors} 筆失敗")
    return results


synthetic_reasoning_data = asyncio.run(main())

# ============================================
#  成本統計
# ============================================

total_input = sum(d["input_tokens"] for d in synthetic_reasoning_data)
total_output = sum(d["output_tokens"] for d in synthetic_reasoning_data)
cost = (total_input * 0.15 + total_output * 0.60) / 1_000_000  # GPT-4o-mini 價格
print(f"總 input tokens:  {total_input:,}")
print(f"總 output tokens: {total_output:,}")
print(f"預估成本（GPT-4o-mini 定價）: ${cost:.4f}")

# ============================================
#  儲存原始生成結果
# ============================================

with open("synthetic_reasoning_raw.jsonl", "w", encoding="utf-8") as f:
    for item in synthetic_reasoning_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n已儲存 synthetic_reasoning_raw.jsonl ({len(synthetic_reasoning_data)} 筆)")
