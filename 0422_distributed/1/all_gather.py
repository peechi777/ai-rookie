def all_gather_py(rank_data):
    result = []
    for sub_list in rank_data:
        for item in sub_list:
            result.append(item)
    return result

if __name__ == "__main__":
    rank_data = [[1],[2],[3],[4]]
    final_result = all_gather_py(rank_data)
    print(rank_data)
    print(final_result)