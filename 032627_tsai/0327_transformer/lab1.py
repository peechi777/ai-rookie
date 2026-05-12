import torch
import matplotlib.pyplot as plt
import numpy as np

def get_sinusoidal_embeddings(n_pos, d_model):
    """
    n_pos: 序列長度 (Max sequence length)
    d_model: 嵌入維度
    """
    # 建立位置向量 pos: (n_pos, 1)
    position = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
    
    # 建立除數項 div_term: (d_model / 2)
    # 只計算偶數項，因為 sin/cos 各佔一半
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    # 初始化 PE 矩陣 (n_pos, d_model)
    pe = torch.zeros(n_pos, d_model)
    
    # 廣播機制
    # 偶數(0, 2, 4)sin
    pe[:, 0::2] = torch.sin(position * div_term)
    # 奇數(1, 3, 5)cos
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# 設定參數
L, D = 100, 128
pe_matrix = get_sinusoidal_embeddings(L, D)


def visualize_pe_similarity(pe):
    # 計算點積矩陣 (L, D) @ (D, L) -> (L, L)
    similarity = torch.matmul(pe, pe.transpose(0, 1))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Position Embedding Dot Product Similarity")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.savefig("position_embedding_similarity.png")
    plt.show()

visualize_pe_similarity(pe_matrix)
