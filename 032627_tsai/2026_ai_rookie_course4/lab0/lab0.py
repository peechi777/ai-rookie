# common_setup.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# 根據需要可以從環境變數覆寫
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def has_cuda():
    return torch.cuda.is_available()  #檢查有無nvidia gpu

def print_env_info():
    print("PyTorch:", torch.__version__)
    print("GPU 可用:", has_cuda())
    if has_cuda():
        print("GPU 名稱:", torch.cuda.get_device_name(0))
        print("GPU RAM(GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

def load_model_and_tokenizer(model_id: str = BASE_MODEL_ID, load_in_4bit: bool = True):
    print(f"載入模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    device_map = "auto"
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if has_cuda() else torch.float32,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model

if __name__ == "__main__":
    print_env_info()
    tokenizer, model = load_model_and_tokenizer()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    message = [
        {
            "role": "system",
            "content": "你是專業客服助理，請用繁體中文，語氣禮貌。"
        },
        {
            "role": "user",
            "content": "最近過得如何？"
        }
    ]
    print("範例訊息:", message)
    #tokenize=false 先看原本message
    chat_template_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    print("Chat template 文字:\n", chat_template_text)
    input_ids = tokenizer(chat_template_text, return_tensors="pt").to(model.device)
    print(input_ids)


    model_output = model.generate(
        **input_ids,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    output_text = tokenizer.decode(model_output[0], skip_special_tokens=True)
    print("模型回應:", output_text)