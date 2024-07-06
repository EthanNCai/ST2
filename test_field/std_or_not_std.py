import numpy as np
from sklearn import preprocessing

vec1 = [1, 3,2]
vec2 = [10, 30, 25]
vec3 = [10, 30, 35]
# 将向量转换为 NumPy 数组
vec1 = np.array(vec1)
vec2 = np.array(vec2)
vec3 = np.array(vec3)


# 创建 MinMaxScaler 对象

def standard_scaled_manhattan_distance(vec1, vec2):
    def scale(time_series):
        scaler = preprocessing.StandardScaler()
        time_series = scaler.fit_transform(np.array(time_series).reshape(-1, 1))
        time_series = time_series.reshape(-1)
        return time_series

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sum(np.abs(scale(vec1) - scale(vec2)))


# 计算曼哈顿距离
dist12 = standard_scaled_manhattan_distance(vec1, vec2)
dist13 = standard_scaled_manhattan_distance(vec1, vec3)


print(f"vec1 和 vec2 之间的曼哈顿距离: {dist12:.5f}")
print(f"vec1 和 vec3 之间的曼哈顿距离: {dist13:.5f}")
