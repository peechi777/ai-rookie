import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp

def ddp_sync_simulation(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size
    )
    my_weight = torch.tensor([10.0])
    my_gradient = torch.tensor([2.0 if rank == 0 else 8.0])  # 模擬不同工作節點的梯度
    lerning_rate = 0.1
    
    dist.all_reduce(my_gradient, op=dist.ReduceOp.SUM)  # 將所有工作節點的梯度相加
    
    my_gradient = my_gradient / world_size
    
    my_weight = my_weight - (lerning_rate * my_gradient)
    print(f"[節點{rank}] 權重: {my_weight.item()}")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    world_size = 2
    mp.spawn(ddp_sync_simulation, args=(world_size,), nprocs=world_size, join=True)