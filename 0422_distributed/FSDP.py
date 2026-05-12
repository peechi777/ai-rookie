import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp

def fsdp_memory_mgmt(rank,ws):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=ws
    )
    my_shard = torch.tensor([float(rank + 1)])  # 模擬每個工作節點的權重碎片
    print(f"[節點{rank}] 靜態: {my_shard.tolist()}")
    
    full_list = [torch.tensor([0.0]) for _ in range(ws)]
    
    dist.all_gather(full_list, my_shard)  # 收集所有工作節點的權重碎片
    
    full_weight = torch.cat(full_list)  
    print(f"[節點{rank}] [配置] 完整: {full_weight.tolist()}")
    
    print(f"[節點{rank}] [配置] 完成")
    
    del full_weight  
    
    try:
        print(full_weight)
    except NameError:
        print(f"[節點{rank}] [回收] 已釋放")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    mp.spawn(fsdp_memory_mgmt, args=(2,), nprocs=2, join=True)
    
    
   