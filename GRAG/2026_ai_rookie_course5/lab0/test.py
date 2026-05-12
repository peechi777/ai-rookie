import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

# vLLM (OpenAI 端點)
os.environ["OPENAI_API_KEY"]  = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:18299/v1"
llm = ChatOpenAI(model="Qwen2.5-3B-Instruct", temperature=0.2)
print(llm.invoke("什麼是 Knowledge Graph").content)

# Neo4j 連線
driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j","password123"))
with driver.session() as s:
    version_info = s.run("CALL dbms.components() YIELD name, versions RETURN name, versions").single()
    print(f"Neo4j Component: {version_info['name']}, Version: {version_info['versions']}")
driver.close()