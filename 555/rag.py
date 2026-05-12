import math

def load_vectors(filename):
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            clean_line = line.replace('[', '').replace(']', '').split(',')
            vectors.append([float(x.strip()) for x in clean_line])
    return vectors

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    norm_v1 = math.sqrt(sum(a**2 for a in v1))
    norm_v2 = math.sqrt(sum(b**2 for b in v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

def main():
    db_vectors = load_vectors('vector_db.txt')
    query_vectors = load_vectors('query.txt')

    query = query_vectors[0]
    k = 4 
    
    results = []
    for i, db_vec in enumerate(db_vectors):
        score = cosine_similarity(query, db_vec)
        results.append({
            "index": i,
            "score": score,
            "vector": db_vec
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    top_k = results[:k]
    
    context_list = []
    for rank, res in enumerate(top_k):
        info = f"文件編號 {res['index']} | 相似度: {res['score']:.4f} | 向量內容: {res['vector']}"
        print(f"Rank {rank+1}: {info}")
        context_list.append(info)
    
    context_str = "\n".join(context_list)

if __name__ == "__main__":
    main()