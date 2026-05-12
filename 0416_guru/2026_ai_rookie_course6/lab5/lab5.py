"""
Lab5 - Prompt 與 Context 消融實驗
==================================
實驗一：完整 prompt vs 極簡 prompt
實驗二：有 context vs 完全不給 context
"""

import os
import json
import random
import time
import asyncio
import requests
import datetime
from tqdm import tqdm
from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#                          設定區
# ==============================================================================

GURU_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output", "question_answer_40chunk_256.json")
LAB4_INFERENCE = os.path.join(SCRIPT_DIR, "..", "lab4", "output", "finetuned_inference.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")

BASE_URL = "http://localhost:18299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"

JUDGE_MODEL_NAME = "gpt-5.1"
GPT_API_URL = ""        # TODO: 填入
GPT_USERNAME = ""       # TODO: 填入
GPT_PASSWORD = ""       # TODO: 填入
GPT_API_BASE = ""       # TODO: 填入

KEY = "RAG_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ==============================================================================
#                          Prompt 定義
# ==============================================================================

# A 組：完整 prompt（與 Lab4 相同）
FULL_SYSTEM = "You are a helpful question answerer who can provide an answer given a question and relevant context."
FULL_USER = """Question: {}
Context: {}

Answer this question using the information given in the context above. Here is things to pay attention to:
- If you need to use CoT (Chain of Thought) reasoning, please do so. If the question is simple and does not require CoT, then do not use it.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
- The response should match the language of the given question.
- End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct."""

# B 組：極簡 prompt
SIMPLE_SYSTEM = "你是一個問答助理。"
SIMPLE_USER = "根據以下內容回答問題。\n\n內容：{}\n\n問題：{}"

# D 組：無 context
NO_CTX_SYSTEM = "You are a helpful question answerer."
NO_CTX_USER = "Please answer the following question. Respond in the same language as the question.\n\nQuestion: {}"


# ==============================================================================
#                          推理函式
# ==============================================================================

def run_inference(guru_data, output_path, system_prompt, user_prompt_template, use_context=True):
    import asyncio
    from openai import OpenAI
    client = OpenAI(api_key="empty", base_url=BASE_URL)
    
    save_data = []
    for data in tqdm(guru_data, desc="推理中"):
        q = data["question"]
        if use_context:
            # 關鍵修正：將 RAG_chunks 轉成字串並截斷長度，避免爆掉 4096 tokens
            context_str = str(data[KEY])
            if len(context_str) > 3500:
                context_str = context_str[:3500] + "..."
            user_msg = user_prompt_template.format(context_str, q)
        else:
            user_msg = user_prompt_template.format(q)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0
        )
        data['predicted_answer'] = response.choices[0].message.content
        save_data.append(data)

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    return save_data


# ==============================================================================
#                          並排比較
# ==============================================================================

def compare_with_lab4(lab4_data, new_data, label, n=5):
    lab4_dict = {d['question']: d.get('predicted_answer', '') for d in lab4_data}
    matched = [d for d in new_data if d['question'] in lab4_dict]
    samples = random.sample(matched, min(n, len(matched)))

    print(f"\n{'='*70}")
    print(f"人工觀察：Lab4（完整） vs {label}（{len(samples)} 筆）")
    print('='*70)

    for i, item in enumerate(samples, 1):
        q = item['question']
        print(f"\n--- 第 {i} 筆 ---")
        print(f"[問題] {q}")
        print(f"[Lab4 完整] {lab4_dict[q][:200]}...")
        print(f"[{label}] {item.get('predicted_answer', '')[:200]}...")


# ==============================================================================
#                          Benchmark
# ==============================================================================

async def run_benchmark(data, output_dir, label, llm):
    from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    correctness = CorrectnessEvaluator(llm=llm)
    batch_runner = BatchEvalRunner({"correctness": correctness}, workers=5, show_progress=True)

    eval_results = await batch_runner.aevaluate_response_strs(
        queries=[str(x['question']) for x in data],
        response_strs=[str(x['predicted_answer']) for x in data],
        reference=[str(x['base_answer']) for x in data]
    )

    scores = [cr.score for cr in eval_results['correctness'] if isinstance(cr.score, (int, float)) and cr.score >= 1.0]
    avg = sum(scores) / max(len(scores), 1)

    print(f"\n[{label}] 平均分: {avg:.2f}/5 ({avg*20:.1f}/100), 筆數: {len(scores)}")

    with open(os.path.join(output_dir, f"{timestamp}_{label}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"{label}: avg={avg:.2f}/5, n={len(scores)}\n")

    return avg


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab5：Prompt 與 Context 消融實驗")
    print("=" * 60)

    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        guru_data = json.load(f)
    with open(LAB4_INFERENCE, 'r', encoding='utf-8') as f:
        lab4_data = json.load(f)

    # 實驗 B：極簡 prompt
    print("\n[實驗 B] 極簡 prompt 推理...")
    simple_path = os.path.join(OUTPUT_DIR, "simple_prompt_inference.json")
    simple_data = run_inference(guru_data, simple_path, SIMPLE_SYSTEM, SIMPLE_USER, use_context=True)
    compare_with_lab4(lab4_data, simple_data, "極簡Prompt", n=5)

    # 實驗 D：無 context
    print("\n[實驗 D] 無 context 推理...")
    no_ctx_path = os.path.join(OUTPUT_DIR, "no_context_inference.json")
    no_ctx_data = run_inference(guru_data, no_ctx_path, NO_CTX_SYSTEM, NO_CTX_USER, use_context=False)
    compare_with_lab4(lab4_data, no_ctx_data, "無Context", n=5)

    # Benchmark
    print("\n[本地評估] 準備使用本地 LLM 進行 Benchmark...")
    from llama_index.llms.openai import OpenAI as LI_OpenAI
    
    # 這裡的 api_base 確保指向你的 vLLM (localhost:8299)
    # api_key 隨便填一個字串即可，因為 vLLM 不會驗證它
    llm = LI_OpenAI(
        model=MODEL_NAME, 
        api_key="empty", 
        api_base=BASE_URL, 
        temperature=0
    )
    llm._get_model_name = lambda: "gpt-3.5-turbo"
    
    async def all_benchmarks():
        s1 = await run_benchmark(simple_data, BENCHMARK_DIR, "simple_prompt", llm)
        s2 = await run_benchmark(no_ctx_data, BENCHMARK_DIR, "no_context", llm)
        return s1, s2

    s1, s2 = asyncio.run(all_benchmarks())

    print(f"\n{'='*60}")
    print("Lab5 消融實驗分數總結")
    print('='*60)
    print(f"  Lab4 完整 prompt + RAG context : （見 Lab4 報告）")
    print(f"  B 組 極簡 prompt + RAG context : {s1:.2f}/5")
    print(f"  D 組 完整 prompt + 無 context  : {s2:.2f}/5")
    print(f"\n請撰寫 ablation_report.md，分析差異原因。")
    print("=" * 60)
