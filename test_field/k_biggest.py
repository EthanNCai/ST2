def find_smallest_k_indices(nums, k):
    if not nums or k <= 0 or k > len(nums):
        return []
    indexed_nums = list(enumerate(nums))
    sorted_indexed_nums = sorted(indexed_nums, key=lambda x: x[1])
    smallest_k_indices = [index for index, value in sorted_indexed_nums[:k]]
    return smallest_k_indices

# 示例使用
nums_list = [3.5, 2.1, 5.9, 1.4, 4.3, 6.7]
k = 3
print(find_smallest_k_indices(nums_list, k))