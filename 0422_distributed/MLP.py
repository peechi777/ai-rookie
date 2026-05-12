import torch
import torch.nn.functional as F
torch.manual_seed(42)
X = torch.randn(4, 8)
A = torch.randn(8, 12)
B = torch.randn(12, 6)
Y_ref = F.relu(X @ A) @ B

A1, A2 = torch.split(A, 6, dim=1)

H1 = F.relu(X @ A1)
H2 = F.relu(X @ A2)

B1, B2 = torch.split(B, 6, dim=0)

Z1 = H1 @ B1
Z2 = H2 @ B2

Y_tp = Z1 + Z2

assert torch.allclose(Y_ref, Y_tp)
print("√ 列切分驗證通過！")
