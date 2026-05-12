from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "password123"))

def check():
    with driver.session() as session:
        print("=== 測試 1：搜尋包含 'Zenith' 的節點 ===")
        res1 = session.run("MATCH (n) WHERE n.name CONTAINS 'Zenith' RETURN n.name as name, labels(n) as labels")
        for r in res1: print(f"Found Node: {r['name']} {r['labels']}")

        print("\n=== 測試 2：搜尋包含 'Smart' 的關係屬性 ===")
        res2 = session.run("MATCH (n)-[r]->(m) WHERE r.item CONTAINS 'Smart' RETURN n.name, type(r), r.item, m.name")
        for r in res2: print(f"Found Relation: ({r[0]})-[:{r[1]} {{item: '{r[2]}'}}]->({r[3]})")

        print("\n=== 測試 3：列出所有關係類型統計 ===")
        res3 = session.run("MATCH ()-[r]->() RETURN type(r), count(r)")
        for r in res3: print(f"Type: {r[0]}, Count: {r[1]}")

check()
driver.close()