import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5])

# 使用concatenate函数合并数组
combined_array = np.concatenate((array1, array2))
print(combined_array)  # 输出: [1 2 3 4 5]