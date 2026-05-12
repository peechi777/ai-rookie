import torch
import torch.nn as nn
import math

class PureTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        
        #FFN 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        
        residual = x
        x = self.ln1(x) #Pre-Norm
        
        # 拆分多頭：(B, L, D) -> (B, L, H, d_k) -> (B, H, L, d_k)
        q = self.w_q(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        
        #K在最後兩維轉置
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        #(B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, D)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        # 殘差連接 1
        x = residual + self.w_out(context)
        
        # Feed-Forward
        residual = x
        x = self.ln2(x) # Pre-Norm
        x = residual + self.ffn(x) #殘差連接 2
        
        return x

#驗證
model = PureTransformerBlock(d_model=128, n_heads=8, d_ff=512)
x = torch.randn(2, 100, 128)
output = model(x)
print(f"最終輸出形狀: {output.shape} (預期為 [2, 100, 128])")