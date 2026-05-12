import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # 第一層線性變換：從 d_model 升維到 d_ff (512 -> 2048)
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二層線性變換：從 d_ff 降維回 d_model (2048 -> 512)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 激活函數
        self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 的維度: (Batch, Seq_Len, d_model)
        
        # 升維
        # (B, L, d_model) -> (B, L, d_ff)
        x = self.activation(self.w_1(x))
        
        #Dropout
        x = self.dropout(x)
        
        # 降維
        # (B, L, d_ff) -> (B, L, d_model)
        x = self.w_2(x)
        
        return x
    
batch_size = 2
seq_len = 100
d_model = 128
d_ff = 512 

# 初始化模型
ffn = PositionWiseFeedForward(d_model, d_ff)

#(Batch, Seq, Dim)
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Forward
output_tensor = ffn(input_tensor)

print(f"輸入維度: {input_tensor.shape}")
print(f"輸出維度: {output_tensor.shape}")
print(f"維度是否一致: {input_tensor.shape == output_tensor.shape}")