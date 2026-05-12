from all_gather import all_gather_py
from reduce_scatter_sum import reduce_scatter_sum_py

def all_reduce_via_two_steps(rank_data):
    partial_gather = reduce_scatter_sum_py(rank_data)
    final_result = all_gather_py(partial_gather)
    return final_result

if __name__ == "__main__":
    rank_data = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]
    ]
    result = all_reduce_via_two_steps(rank_data)
    
    print(rank_data)
    for i in range(len(rank_data)):
        print(f"rank {i}: {result}")
