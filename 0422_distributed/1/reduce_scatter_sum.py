def reduce_scatter_sum_py(rank_data):
    
    world_size = len(rank_data)    #要分幾次
    data_len = len(rank_data[0])   #要加幾次
    
    sum_result = [0] * data_len
    for sub_list in rank_data:
        for i in range(data_len):
            sum_result[i] += sub_list[i]
            
    final_outputs = []
    for rank in range(world_size):
        final_outputs.append([sum_result[rank]])
    return final_outputs

if __name__ == "__main__":
    rank_data = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]
    ]
    result = reduce_scatter_sum_py(rank_data)
    for i,res in enumerate(result):
        print(f"rank {i}: {res}")