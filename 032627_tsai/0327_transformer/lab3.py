import torch
import torch.nn.functional as F
import math

B, L, D = 1, 4, 8  
d_k = D

# 模擬 Q, K 的輸入
q = torch.randn(B, L, D)
k = torch.randn(B, L, D)

# 計算原始 (B, L, L)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

# Causal Mask 
# 下三角為1
mask = torch.tril(torch.ones(L, L)) 
# 0替換成-inf，1的替換成0
mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

#Softmax
masked_scores = scores + mask
attn_weights = F.softmax(masked_scores, dim=-1)

print("--- Mask 矩陣 ---")
print(mask)
print("\n--- Attention Weights (Softmax 後) ---")
print(attn_weights[0]) 