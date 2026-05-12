import torch
torch.manual_seed(42)
X = torch.randn(4, 8)
A = torch.randn(8, 6)
Y_ref = X @ A

A1, A2 = torch.split(A, 3, dim=1)

Y1 = X @ A1
Y2 = X @ A2

Y_tp = torch.cat([Y1, Y2], dim=1)

assert torch.allclose(Y_ref, Y_tp)
print("√ 列切分驗證通過！")