import os
import json
import random
import time
import asyncio
import datetime
from tqdm import tqdm
from openai import AsyncOpenAI
# 修正 LlamaIndex 相關匯入
from llama_index.core.evaluation import CorrectnessEvaluator, BatchEvalRunner

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#                          設定區
# ==============================================================================
GURU_OUTPUT = os.path.join(SCRIPT_DIR, "..", "lab1", "output", "question_answer_40chunk_256.json")
BASELINE_INFERENCE = os.path.join(SCRIPT_DIR, "..", "lab2", "output", "baseline_inference.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BENCHMARK_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")

BASE_URL = "http://localhost:18299/v1"
MODEL_NAME = "Qwen2.5-3B-Instruct"
GPT_API_BASE = "http://localhost:18299/v1"
JUDGE_MODEL_NAME = "Qwen2.5-3B-Instruct" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ==============================================================================
#                          Prompt 模板
# ==============================================================================
RAG_SYSTEM_PROMPT = "You are a helpful question answerer who can provide an answer given a question and relevant context."
RAG_USER_PROMPT = """Question: {}
Context: {}

Answer this question using the information given in the context above. Pay attention to:
- Use CoT (Chain of Thought) if necessary.
- Include quotes in ##begin_quote## and ##end_quote## if copying from context.
- Match the language of the question.
- End with <ANSWER>: $answer"""

# ==============================================================================
#                          Part 1：Finetuned 推理
# ==============================================================================
# ==============================================================================
#                          Part 1：Finetuned 推理 (修正版)
# ==============================================================================

async def process_single_request(client, data, pbar):
    """處理單一請求的非同步邏輯"""
    context_text = "\n\n".join(data["chunk"])
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": RAG_USER_PROMPT.format(data["question"], context_text)}
            ],
            n=1,
            top_p=1,
            temperature=0
        )
        data['predicted_answer'] = response.choices[0].message.content
        data['test_chunk'] = "context"
    except Exception as e:
        print(f"Error processing question: {data.get('question', 'Unknown')[:20]}... | Error: {e}")
        data['predicted_answer'] = f"ERROR: {str(e)}"
    finally:
        pbar.update(1)
    return data

def run_finetuned_inference(guru_data, output_path):
    """使用原生 AsyncOpenAI 進行併發推理"""
    print(f"[Finetuned 推理] 端點: {BASE_URL}")
    print(f"[Finetuned 推理] 模型: {MODEL_NAME}")
    print(f"[Finetuned 推理] 資料筆數: {len(guru_data)}")

    async def main_inference():
        client = AsyncOpenAI(api_key="empty", base_url=BASE_URL, timeout=1200)
        
        # 使用 Semaphore 控制併發數 (限制為 10，避免壓垮本地 vLLM)
        sem = asyncio.Semaphore(10)
        
        async def sem_task(data, pbar):
            async with sem:
                return await process_single_request(client, data, pbar)

        with tqdm(total=len(guru_data), desc="Finetuned 推理") as pbar:
            tasks = [sem_task(data, pbar) for data in guru_data]
            results = await asyncio.gather(*tasks)
        return results

    # 執行非同步任務
    save_data = asyncio.run(main_inference())

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"[Finetuned 推理] 完成: {output_path}")
    return save_data

# ==============================================================================
#                          Part 2：人工觀察
# ==============================================================================
def side_by_side_comparison(baseline_data, finetuned_data, n=5):
    base_dict = {d['question']: d.get('predicted_answer', '') for d in baseline_data}
    matched = [d for d in finetuned_data if d['question'] in base_dict]
    samples = random.sample(matched, min(n, len(matched)))

    print(f"\n{'='*70}\n人工觀察：Base vs Finetuned 並排比較\n{'='*70}")
    for i, item in enumerate(samples, 1):
        print(f"\n--- 第 {i} 筆 ---\n[問題] {item['question']}")
        print(f"[Base Model] {base_dict.get(item['question'], '')[:150]}...")
        print(f"[Finetuned]  {item.get('predicted_answer', '')[:150]}...")

# ==============================================================================
#                          Part 3：Benchmark
# ==============================================================================
async def run_benchmark(data, output_dir, model_name, llm, label="Finetuned Model"):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    correctness = CorrectnessEvaluator(llm=llm)
    batch_runner = BatchEvalRunner({"correctness": correctness}, workers=4, show_progress=True)

    print(f"\n[Benchmark] 評估 {label}...")
    start = time.time()
    
    eval_results = await batch_runner.aevaluate_response_strs(
        queries=[str(x['question']) for x in data],
        response_strs=[str(x.get('predicted_answer', '')) for x in data],
        reference=[str(x.get('base_answer', '')) for x in data]
    )
    
    output_datas = []
    for cr in eval_results['correctness']:
        output_datas.append({
            "question": cr.query,
            "rating": cr.score,
            "feedback": cr.feedback
        })

    scores = [d['rating'] for d in output_datas if d['rating'] is not None]
    avg = sum(scores) / max(len(scores), 1)
    print(f"\n>>> {label} 平均分數: {avg:.2f} / 5 ({avg*20:.1f} / 100)")

    # 存檔
    tag = label.lower().replace(" ", "_")
    with open(os.path.join(output_dir, f"{timestamp}_{tag}.json"), 'w', encoding='utf-8') as f:
        json.dump(output_datas, f, ensure_ascii=False, indent=4)

# ==============================================================================
#                          主程式
# ==============================================================================
if __name__ == "__main__":
    # 1. 讀取 Lab1 資料
    with open(GURU_OUTPUT, 'r', encoding='utf-8') as f:
        guru_data = json.load(f)

    # 2. 推理
    ft_path = os.path.join(OUTPUT_DIR, "finetuned_inference.json")
    ft_data = run_finetuned_inference(guru_data, ft_path)

    # 3. 並排比較
    with open(BASELINE_INFERENCE, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    side_by_side_comparison(baseline_data, ft_data, n=5)

    # 4. LlamaIndex 評測
    from llama_index.llms.openai import OpenAI

# 1. 初始化，model 名稱用 Qwen，這樣對 vLLM 才不會 404
    llm = OpenAI(model="Qwen2.5-3B-Instruct", api_base=BASE_URL, api_key="empty")

# 2. 暴力破解：直接把那個會報錯的函數換成一個永遠回傳固定數值的函數
# 這樣它就不會去跑名稱比對，也不會噴 ValueError
    llm._get_model_name = lambda: "gpt-3.5-turbo"

    asyncio.run(run_benchmark(baseline_data, BENCHMARK_DIR, JUDGE_MODEL_NAME, llm, label="Base Model"))
    asyncio.run(run_benchmark(ft_data, BENCHMARK_DIR, JUDGE_MODEL_NAME, llm, label="Finetuned Model"))