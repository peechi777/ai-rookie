import torch
import torch.nn as nn
torch.manual_seed(42)
X = torch.randn(4, 8)
target = torch.randn(4, 2)
layers = [nn.Linear(8,6),nn.Linear(6,6),nn.Linear(6,4),nn.Linear(4,2)]
Y_ref =  X
for layer in layers:
    Y_ref = torch.relu(layer(Y_ref))

stage_0 = layers[0]
stage_1 = layers[1]
stage_2 = layers[2]
stage_3 = layers[3]

act_0 = torch.relu(stage_0(X))
act_1 = torch.relu(stage_1(act_0))
act_2 = torch.relu(stage_2(act_1))
Y_tp = stage_3(act_2)

assert torch.allclose(Y_ref, Y_tp)
print("√ Pipeline 前向傳遞驗證通過！")
print(f"Stage 0 參數數量: {sum(p.numel() for p in stage_0.parameters() if p.requires_grad)}")
print(f"Stage 1 參數數量: {sum(p.numel() for p in stage_1.parameters())}")
print(f"Stage 2 參數數量: {sum(p.numel() for p in stage_2.parameters())}")
print(f"Stage 3 參數數量: {sum(p.numel() for p in stage_3.parameters())}")
