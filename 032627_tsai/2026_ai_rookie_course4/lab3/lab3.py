# lab3_data_cleaning.py
import json
import random
import re
import unicodedata
import hashlib
from urllib import response
# import requests
from typing import Any, Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# TODO 0: 嘗試匯入 opencc（如無則 fallback）
try:
    from opencc import OpenCC
    _cc_s2t = OpenCC("s2t")
    def to_trad(s: str) -> str:
        return _cc_s2t.convert(s)
except Exception:
    def to_trad(s: str) -> str:
        return s

BADWORDS = ("垃圾", "白癡", "去死")  # 示意

def normalize_text(s: str) -> str:
    # TODO 1: 實作基礎清洗流程
    s = to_trad(s)      # 1) 簡轉繁
    s = unicodedata.normalize("NFKC",s)     # 2) Unicode 正規化
    s = re.sub(r"\s", " ",s).strip()    # 3) 去掉多餘空白
    return s

def is_toxic(s: str) -> bool:
    # TODO 2: 如字串含 BADWORDS 中的任何詞就視為 toxic
    return any(word in s for word in BADWORDS)

"""
def call_llm(messages):
    data = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.6,
        "stream": False,
        "max_tokens": 1024
    }
    
    response = requests.post("http://127.0.0.1:8299/v1/chat/completions", json=data)
    response = response.json()
    response_text = response["choices"][0]["message"]["content"]
    return response_text
"""

from more_topics import more_topics
def build_synthetic_examples(n: int = 60) -> List[Dict[str, Any]]:
    """
    建立一個模擬的客服訓練資料集，
    每筆資料格式：
    {
      "id": "ex0001",
      "messages": [ {role, content}, ... ],
      "topic": "退貨流程",
      "language": "zh"
    }
    """
    topics = [
        ("查詢出貨", "請幫我查詢訂單出貨進度，訂單號碼是 ABC123。"),
        ("退貨流程", "我想辦退貨，該怎麼做？"),
        ("退款時程", "退款通常需要幾天會到帳？"),
        ("修改地址", "下單後可以修改收件地址嗎？"),
        ("維修保固", "產品有保固嗎？如何送修？"),
        ("發票補發", "可以補發發票嗎？流程是？"),
    ]

    def synth_assistant(topic: str, user: str) -> str:
                # TODO 3: 回傳一個固定但看起來合理的客服回答（繁體中文）
        assistant_content = f"""顧客您好:
    感謝您聯繫我們關於{topic}的問題。針對您的詢問："{user}"，我們建議您按照以下步驟進行：
    1. 請先登入系統並確認您的「{topic}」相關權限是否已開啟。
    2. 依照頁面提示步驟，將您的具體需求填寫完整並點選「提交」。
    3. 提交後請保留單號，我們的專員將在 1 個工作天內主動與您聯繫。
    如果您在操作過程中遇到任何困難，請隨時與我們聯繫。我們的客服團隊將竭誠為您提供協助。
    祝您有美好的一天！
    客服團隊 敬上"""
        return assistant_content

    data = []
    for i in tqdm(range(n)):
        topic, user = random.choice(more_topics)
        # TODO 4: 生成單輪對話
        assistant_content = synth_assistant(topic, user)

        messages = [
            {"role": "system", "content": "你是專業客服助理，請用繁體中文，語氣禮貌。"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content}
        ]
        data.append(
            {
                "id": f"ex{i:04d}",
                "messages": messages,
                "topic": topic,
                "language": "zh-Hant",
            }
        )
    return data

def clean_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_user_len: int = 512,
    max_total_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """
    - 對 messages 的 content 做 normalize / 繁簡轉換 / 毒性過濾
    - 過短/過長樣本刪除
    - 以 user 內容的 hash 去重
    - 用 chat template 估計 token 長度，超過 max_total_tokens 的刪除
    """
    cleaned = []
    seen_keys = set()

    for ex in tqdm(examples, desc="Cleaning"):
        # TODO 6: 遍歷 messages，做 normalize_text & is_toxic 檢查
        # 若發現毒性，整筆丟棄
        new_messages = []
        is_bad = False
        user_content_parts = []
        
        for msg in ex["messages"]:
            content = msg["content"]
        
            if msg["role"] == "user":
                content = normalize_text(content) # User 徹底清洗
                user_content_parts.append(content)
            else:
                content = to_trad(content).strip() # Assistant 保留換行
            
            if is_toxic(content):
                is_bad = True
                break
            
            new_messages.append({"role": msg["role"], "content": content})

        if is_bad: continue

        # TODO 7: 計算所有 user content 的總長度，過短/過長丟棄
        # user_concat = ...
        user_concat = "".join(user_content_parts)
        if len(user_concat) < 5 or len(user_concat) > max_user_len:
            continue

        # TODO 8: 用 chat_template 估 token 長度，超過 max_total_tokens 丟棄
        # chat_tokens = ...
        token_ids = tokenizer.apply_chat_template(new_messages, add_generation_prompt=False)
        if len(token_ids) > max_total_tokens:
            continue
        # TODO 9: 用 user_concat 做 hash，做去重
        # key = hashlib.md5(user_concat.encode("utf-8")).hexdigest()
        # 若 key 已存在於 seen_keys 則跳過
        key = hashlib.md5(user_concat.encode("utf-8")).hexdigest()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # 否則加入 cleaned
        ex["messages"] = new_messages
        cleaned.append(ex)
    return cleaned

def save_json(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    raw_examples = build_synthetic_examples(n=1000)
    cleaned = clean_dataset(raw_examples, tokenizer, max_user_len=300, max_total_tokens=1024)
    random.shuffle(cleaned)

    n = len(cleaned)
    train = cleaned[: int(0.9 * n)]
    test = cleaned[int(0.9 * n) : n]

    save_json("train.json", train)
    save_json("test.json", test)

    print(f"資料筆數: train={len(train)}, test={len(test)}")

if __name__ == "__main__":
    main()