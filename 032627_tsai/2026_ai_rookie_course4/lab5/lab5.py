# lab5_inference_and_eval.py
import json
import re
from collections import Counter
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def load_base_and_adapter(base_model_id: str, adapter_dir: str):
    """
    載入基礎模型 + LoRA 權重。
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model_id, use_fast=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model

def generate_reply(tokenizer, model, messages, max_new_tokens: int = 256):
    """
    使用正確的 chat template 做推理。
    """
    # TODO 1: 用 apply_chat_template 組成 prompt（add_generation_prompt=True）
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # TODO 2: tokenize + model.generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用 Greedy Search，每次選機率最高的字，結果最穩定
            repetition_penalty=1.1
        )
    
    input_len = inputs["input_ids"].shape[1]
    reply_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)    

    return reply_text.strip()

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
"""
def evaluate_one(example: Dict[str, Any], reply: str) -> Dict[str, Any]:
    
    啟發式評估一條回覆：
    - 有沒有禮貌用語
    - 有沒有步驟/結構字樣
    - 中文比例是否夠高
    - 是否包含 topic/關鍵詞
    
    # TODO 3: 自行設計幾個簡單的規則打分
    polite = False
    structured = False
    zh_ok = False
    topical = False

    score = sum([polite, structured, zh_ok, topical]) / 4.0
    errors = []
    if not polite:
        errors.append("缺少禮貌用語")
    if not structured:
        errors.append("缺少結構化步驟")
    if not zh_ok:
        errors.append("語言非中文為主")
    if not topical:
        errors.append("主題相關性不足")

    return {"score": score, "errors": errors, "reply": reply}
"""

def evaluate_one(example: Dict[str, Any], reply: str) -> Dict[str, Any]:
    """
    啟發式評估一條回覆：
    """
    # TODO 3: 自行設計幾個簡單的規則打分
    # 1. Polite: 是否包含常見禮貌詞彙
    polite_keywords = ["您好", "請", "謝謝", "協助", "抱歉", "祝您"]
    polite = any(word in reply for word in polite_keywords)

    # 2. Structured: 檢查是否有列點或步驟字眼
    structured = bool(re.search(r"(\d\.|\d\)|步驟|首先|接下來|最後)", reply))

    # 3. zh_ok: 檢查中文字元比例是否過半
    zh_chars = len(CJK_RE.findall(reply))
    total_chars = len(reply) + 1e-9
    zh_ok = (zh_chars / total_chars) > 0.5

    # 4. Topical: 檢查是否包含 example 中的關鍵主題詞
    topic = example.get("topic", "")
    topical = topic.lower() in reply.lower() if topic else True

    # 計算分數與錯誤
    errors = []
    if not polite: errors.append("缺少禮貌用語")
    if not structured: errors.append("缺少結構化步驟")
    if not zh_ok: errors.append("語言非中文為主")
    if not topical: errors.append("主題相關性不足")

    score = sum([polite, structured, zh_ok, topical]) / 4.0

    return {"score": score, "errors": errors, "reply": reply}

def main():
    tokenizer, model = load_base_and_adapter(BASE_MODEL_ID, "adapter")

    # TODO 4: 讀取 test.jsonl
    test_examples = []
    try:
        with open("workdir/test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                test_examples.append(json.loads(line))
    except FileNotFoundError:
        print("錯誤：找不到 workdir/test.jsonl 檔案。")
        return

    results = []
    total_score = 0.0
    err_counter = Counter()
    
    for ex in test_examples:
        messages = ex["messages"]
        reply = generate_reply(tokenizer, model, messages, max_new_tokens=256)
        eval_res = evaluate_one(ex, reply)
        eval_res["id"] = ex.get("id", "N/A")
        
        results.append(eval_res)
        total_score += eval_res["score"]
        err_counter.update(eval_res["errors"])

    # TODO 5: 計算平均分數與錯誤統計
    avg_score = total_score / len(test_examples) if test_examples else 0.0

    print("-" * 30)
    print(f"平均分數: {avg_score:.3f}")
    print("錯誤統計：", dict(err_counter))
    print("-" * 30)

    # TODO 6: 輸出 eval_results.json
    output_report = {
        "summary": {
            "average_score": avg_score,
            "error_distribution": dict(err_counter),
            "total_count": len(test_examples)
        },
        "results": results
    }

    with open("workdir/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output_report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()