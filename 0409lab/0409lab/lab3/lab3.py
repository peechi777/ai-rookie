import json
from typing import List
from neo4j import GraphDatabase
import requests

driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j","password123"))

def call_llm(messages):
    data = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.6,
        "stream": False,
        "max_tokens": 1024
    }
    
    response = requests.post("http://127.0.0.1:18299/v1/chat/completions", json=data)
    response = response.json()
    response_text = response["choices"][0]["message"]["content"]
    return response_text

def extract_entities(question:str)->List[str]:
    system_prompt = """你是一位專業的助手。
請從問題中找出實體。

**特別規則：**
1. 如果使用者提到「這家公司」或「該公司」，且沒有指名道姓，請根據上下文推斷或回傳圖譜核心實體 ["Acme"]。
2. 僅回傳 JSON 格式。

範例：
問題：這家公司有什麼產品？
回傳：{"entities": ["Acme"]}
"""
    user_prompt = f"問題：{question}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = call_llm(messages)
    try:
        data = json.loads(response.strip())
        return data.get("entities",[])
    except Exception:
        return []

def fetch_subgraph(entities:List[str], max_hop:int=1):
    if not entities: return []

    query = """ 
    MATCH p=(n)-[*1..{k}]-(m) 
    WHERE n.name IN $ents
    RETURN p LIMIT 50
    """.format(k = max_hop)
    
    with driver.session() as s:
        records=s.run(query,ents=entities)
        triples=[]
        for r in records:
            for rel in r["p"].relationships:
                triples.append(f"({rel.start_node['name']})-[:{rel.type}]->({rel.end_node['name']})")
        return list(set(triples))

def qa_graph(question:str):
    entities=extract_entities(question)
    triples=fetch_subgraph(entities)
    context="\n".join(triples) if triples else "（查無相關圖譜）"
    # TODO 2: 撰寫 prompt，將圖譜三元組（context）當作上下文，讓 LLM 根據這些關係回答問題
    # 提示：prompt 應包含：(1) 已知的圖譜關係 (2) 使用者的問題 (3) 要求 LLM 根據上下文回答
    prompt = f"""
你是一位專業的知識圖譜分析師。
以下是從圖資料庫中檢索出的相關實體關係（三元組格式）：
---------------------
{context}
---------------------

請根據上述提供的圖譜關係回答使用者的問題。
如果根據上述資訊無法回答，請回答「根據目前的知識圖譜，我無法回答這個問題」。

使用者問題：{question}
請用簡潔專業的語氣回答：
"""  
    
    
    # <-- 請撰寫你的 prompt（用 f-string 嵌入 context 和 question）
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = call_llm(messages)
    answer = response.strip()
    return answer, triples

if __name__=="__main__":
    while True:
        q=input("提問(Enter 離開)：")
        if not q: break
        ans,ev=qa_graph(q)
        print("Answer:",ans)
        print("Evidence triples:",ev)