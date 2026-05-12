import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="AI 客服推理腳本")
    parser.add_argument("--user", type=str, required=True, help="用戶問題")
    parser.add_argument("--system", type=str, default="你是專業客服助理，請用繁體中文，語氣禮貌。", help="系統提示詞")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="adapter")
    args = parser.parse_args()

    print(f"載入模型中...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    base = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_cfg, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("-" * 20)
    print(f"用戶問題: {args.user}")
    print(f"AI 回覆: ")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip())
    print("-" * 20)

if __name__ == "__main__":
    main()
