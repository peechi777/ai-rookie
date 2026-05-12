# lab6_ablation_and_packaging.py
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 2. 現在可以使用絕對導入，不要加 ".."
try:
    from lab5.lab5 import evaluate_one
except ImportError:
    print("  ")
    sys.exit(1)

_LAB6_DIR = Path(__file__).resolve().parent
_WORKDIR = _LAB6_DIR / "workdir"

def load_model_for_inference(base_model_id: str, adapter_dir: str):
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

def generate_correct(tokenizer, model, messages) -> str:
    """
    正確模板：使用 apply_chat_template + system。
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,        #Greedy Search 
            repetition_penalty=1.1
        )
    
    input_len = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return reply.strip()

def generate_wrong(tokenizer, model, messages) -> str:
# TODO 2: 錯誤模板實作 (故意不用 chat template)
    user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    # 模擬新手直接拼接字串的做法
    prompt = f"請用繁體中文回答：\n{user_text}\n回答："
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False
        )
    
    # 因為沒有明確的生成引導標籤，通常會連同 Prompt 一起解碼或效果極差
    input_len = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return reply.strip()

def run_template_ablation(test_examples: List[Dict[str, Any]], tokenizer, model, max_samples: int = 5):
    """
    對前幾筆測試資料做正確模板 vs 錯誤模板的比較。
    """
    for ex in test_examples[:max_samples]:
        good = generate_correct(tokenizer, model, ex["messages"])
        bad = generate_wrong(tokenizer, model, ex["messages"])

        eval_good = evaluate_one(ex, good)
        eval_bad = evaluate_one(ex, bad)

        print(f"\n[{ex['id']}] 正確模板 分數={eval_good['score']:.2f}, 錯誤模板 分數={eval_bad['score']:.2f}")
        print("正確模板回覆：", textwrap.shorten(good, width=200, placeholder=" ..."))
        print("錯誤模板回覆：", textwrap.shorten(bad, width=200, placeholder=" ..."))

def write_inference_script(base_model_id: str, adapter_dir: str, path: str):
    """
    產生一個簡單的推理腳本 inference.py，以方便 CLI 使用。
    """
    # TODO 3: 寫入一段 Python 程式碼到 path
    # 內容可包含 argparse 參數：--model_id, --adapter_dir, --system, --user
    code = f'''\
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="AI 客服推理腳本")
    parser.add_argument("--user", type=str, required=True, help="用戶問題")
    parser.add_argument("--system", type=str, default="你是專業客服助理，請用繁體中文，語氣禮貌。", help="系統提示詞")
    parser.add_argument("--model_id", type=str, default="{base_model_id}")
    parser.add_argument("--adapter_dir", type=str, default="adapter")
    args = parser.parse_args()

    print(f"載入模型中...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    base = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_cfg, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {{"role": "system", "content": args.system}},
        {{"role": "user", "content": args.user}}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("-" * 20)
    print(f"用戶問題: {{args.user}}")
    print(f"AI 回覆: ")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip())
    print("-" * 20)

if __name__ == "__main__":
    main()
'''
# TODO: 在這裡放入實際推理程式碼骨架

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def write_readme(base_model_id: str, path: str):
    # TODO 4: 產生 README_delivery.txt
    readme = f"""\
========================================
SFT 指令微調模型交付說明 (Customer Service AI)
========================================

1. 基礎模型資訊
   - Base Model: {base_model_id}
   - 微調技術: QLoRA (4-bit Quantization)

2. 檔案目錄
   - adapter/ : 包含 LoRA 權重與 tokenizer 配置文件
   - inference.py : 獨立推理腳本
   - test.jsonl : 原始測試資料集

3. 執行環境
   - 安裝依賴: uv sync
   - 硬體需求: NVIDIA GPU (建議 6GB VRAM 以上)

4. 推理指令範例
   python inference.py --user "我想修改我的收件地址，請問該怎麼操作？" --adapter_dir "adapter"

5. 注意事項
   - 務必使用 tokenizer.apply_chat_template 進行推理，否則回答品質會大幅下降。
   - 本模型已針對禮貌用語與結構化回答進行優化。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)

def main():
    adapter_dir = str(_WORKDIR / "adapter")
    tokenizer, model = load_model_for_inference(BASE_MODEL_ID, adapter_dir)

    # 讀取 test.jsonl
    test_examples = []
    with open(_WORKDIR / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    # TODO 5: 執行模板消融實驗
    run_template_ablation(test_examples, tokenizer, model, max_samples=5)

    # TODO 6: 寫出 inference.py 與 README_delivery.txt
    write_inference_script(BASE_MODEL_ID, adapter_dir, str(_WORKDIR / "inference.py"))
    write_readme(BASE_MODEL_ID, str(_WORKDIR / "README_delivery.txt"))

    print("已輸出 workdir/inference.py 與 workdir/README_delivery.txt")

if __name__ == "__main__":
    main()