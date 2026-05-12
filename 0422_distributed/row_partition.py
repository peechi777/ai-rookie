import torch
torch.manual_seed(42)
X = torch.randn(4, 8)
A = torch.randn(8, 6)
Y_ref = X @ A

A1, A2 = torch.split(A, 4, dim=0)

X1, X2 = torch.split(X, 4, dim=1)

Z1 = X1 @ A1
Z2 = X2 @ A2

Y_tp =  Z1 + Z2

assert torch.allclose(Y_ref, Y_tp)
print("√ 列切分驗證通過！")