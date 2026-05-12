import torch
import time
import matplotlib.pyplot as plt
from lab5 import KVCacheTransformerBlock # 假設你之前的類別存在 lab5.py

def run_performance_test(max_len=512):
    D, H = 128, 8
    block = KVCacheTransformerBlock(D, H, 512).eval()
    
    full_times = []
    cache_times = []
    
    # 模擬輸入
    x_full = torch.randn(1, 1, D) #warm-up
    
    print("全量重算")
    for i in range(1, max_len + 1): #讀入長度從1到max_len
        input_seq = torch.randn(1, i, D)
        start = time.time()
        # 每次都丟入長度為 i 的序列
        _ = block(input_seq)
        full_times.append(time.time() - start)

    print("KV Cache")
    past_kv = None #放筆記
    for i in range(1, max_len + 1):
        single_token = torch.randn(1, 1, D) #每次只丟入 1 個 token
        start = time.time()
        _, past_kv = block(single_token, past_kv=past_kv) #筆記更新
        cache_times.append(time.time() - start)

    return full_times, cache_times


full, cache = run_performance_test(512)

plt.figure(figsize=(10, 6))
plt.plot(full, label='Full Recalculation (Quadratic)', color='red')
plt.plot(cache, label='KV Cache (Linear)', color='blue')
plt.xlabel('Sequence Length')
plt.ylabel('Time per Token (s)')
plt.title('Inference Speed: Full vs KV Cache')
plt.legend()
plt.grid(True)
plt.savefig('lab6.png') 
