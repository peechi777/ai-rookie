import torch
import torch.nn as nn
torch.manual_seed(42)
X = torch.randn(8, 6)
target = torch.randn(8, 2)
stage_1 = nn.Linear(6,4)
stage_2 = nn.Linear(4,2)

Y_ref =  stage_2(torch.relu(stage_1(X)))
loss_ref = nn.MSELoss()(Y_ref, target)

M = 4

X_micros = torch.chunk(X, M, dim=0)
T_micros = torch.chunk(target, M, dim=0)

total_loss = torch.tensor(0.0)

for i in range(M):
    h = torch.relu(stage_1(X_micros[i]))
    out = stage_2(h)
    
    micro_loss = nn.MSELoss()(out, T_micros[i])
    total_loss += (micro_loss)/M    
    
print(f"Gpipe loss: {total_loss.item():.4f}")