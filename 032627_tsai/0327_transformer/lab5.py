import torch
import torch.nn as nn
import math

class KVCacheTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.h = n_heads # 頭數
        self.d_k = d_model // n_heads # 每頭的維度
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)    # LayerNorm 1

    def forward(self, x, past_kv=None):
        B, L_new, D = x.shape 
        x_norm = self.ln1(x)
        
        # (B, H, L_new, d_k)
        q = self.w_q(x_norm).view(B, L_new, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(x_norm).view(B, L_new, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(x_norm).view(B, L_new, self.h, self.d_k).transpose(1, 2)

        if past_kv is not None:
            k_old, v_old = past_kv
            k = torch.cat([k_old, k], dim=2) # 將新的 K 拼接在舊的 K 後面
            v = torch.cat([v_old, v], dim=2) # 將新的 V 拼接在舊的 V 後面
        
        present_kv = (k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1) # (B, H, L_new, L_total)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L_new, D)
        return self.w_out(out), present_kv


def greedy_decode_demo():
    B, D, H = 1, 128, 8
    block = KVCacheTransformerBlock(D, H, 512)
    block.eval()

    tokens = torch.randn(B, 3, D) # 模擬前面已經生成了 3 個 token 的輸入
    
    out_full, _ = block(tokens)
    next_token_gen_a = out_full[:, -1:, :] 

    _, cache = block(tokens[:, :2, :]) 
    out_cache, cache_final = block(tokens[:, 2:3, :], past_kv=cache) # 使用前兩個 token 的 K/V 作為 cache，生成下一個 token 的輸出
    next_token_gen_b = out_cache 

    diff = torch.abs(next_token_gen_a - next_token_gen_b).max().item()
    print(f"誤差: {diff:.8e}")
    print(f"Cache K 維度: {cache_final[0].shape}")

if __name__ == "__main__":
    greedy_decode_demo()