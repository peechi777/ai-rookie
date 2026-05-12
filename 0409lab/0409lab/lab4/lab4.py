import os, re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ------- env & clients -------
os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["OPENAI_API_BASE"]="http://localhost:18299/v1"
LLM_MODEL="Qwen2.5-3B-Instruct"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Vector store（與 Lab 1 相同索引：SemanticChunker 建於 lab1/chroma_store）
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_chroma_dir = Path(__file__).resolve().parent.parent / "lab1" / "chroma_store"
vectordb = Chroma(persist_directory=str(_chroma_dir), embedding_function=emb)

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j","password123"))

# ------- functions ----------
def candidate_entities(question:str, k:int=4):
    # TODO 1: 用向量搜尋找出候選實體
    # 步驟：
    #   1. 使用 vectordb.similarity_search(question, k=k) 取得相關文件
    #   2. 用 re.findall(r'[A-Z][A-Za-z]+', d.page_content) 從每份文件中提取大寫開頭的詞
    #   3. 收集到 set 中去重，最後回傳前 5 個
    docs = vectordb.similarity_search(question, k=k)
    text_concat = " ".join([d.page_content for d in docs])
    candidates = set(re.findall(r'[A-Z][A-Za-z]+', text_concat))
    
    unique_ents = list(set(candidates))[:5]
    return unique_ents

def graph_expand(ents, hop=2):
    if not ents: return [] 
    
    query = f"""
    MATCH p=(n)-[*1..{hop}]-(m)
    WHERE n.name IN $ents
    RETURN p LIMIT 50
    """
    triples = []
    with driver.session() as session:
        result = session.run(query, ents=ents)
        for record in result:
            # 從路徑 p 中取出所有關係
            path = record["p"]
            for rel in path.relationships:
                # 格式化為三元組字串
                s = rel.start_node['name']
                t = rel.type
                o = rel.end_node['name']
                triples.append(f"({s})-[:{t}]->({o})")
    
    # 去重後回傳
    return list(set(triples))

def answer_with_graph(question:str):
    ents=candidate_entities(question)
    triples=graph_expand(ents)
    context="\n".join(triples) if triples else "（圖譜中找不到相關關係）"
    # TODO 3: 撰寫 prompt，讓 LLM 根據圖譜三元組回答問題
    # 要求：
    #   - 角色設定為「企業知識專家」
    #   - 只能根據圖譜關係（context）回答
    #   - 若資訊不足回答「無足夠資訊」
    #   - 以繁體中文回答
    # <-- 請撰寫你的 prompt（用 f-string 嵌入 context 和 question）
    prompt = f"""
你是一位專業的「企業知識專家」。請根據以下提供的知識圖譜三元組，
以「繁體中文」回答使用者的問題。

[知識圖譜上下文]
{context}

[使用者問題]
{question}

請注意：
1. 僅根據上述圖譜資訊回答。
2. 如果資訊不足，請誠實說明無法回答。
3. 回答需簡潔且具邏輯性。
"""
    return llm.invoke(prompt).content.strip(), triples, ents

if __name__=="__main__":
    while True:
        q=input("提問 (Enter 離開)：")
        if not q: break
        ans,triples,ents=answer_with_graph(q)
        print("候選實體：",ents)
        print("Evidence triples:",triples)
        print("Answer:",ans,"\n")