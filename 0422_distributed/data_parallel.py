import torch

main_weight = torch.tensor([10.0]) #主權重
learning_rate = 0.1

worker_gradients = [
    torch.tensor([2.0]),
    torch.tensor([4.0]),
    torch.tensor([6.0]),
    torch.tensor([8.0])
]

avg_grad = sum(worker_gradients) / len(worker_gradients)

main_weight = main_weight - (learning_rate * avg_grad)

print(f"[主節點] 更新完成 權重: {main_weight.item()}")