import matplotlib.pyplot as plt
import numpy as np

# 输入数字n
n = 15  # 例如，n = 10

# 计算衰减率b
b = np.log(1/n) / (n - 1)
b *= 1.5

# 生成指数衰减序列
x_values = np.arange(n)  # x值从0到n-1
y_values = np.exp(-b * x_values)  # y值是指数衰减序列

# 确保序列的和为1
total_sum = np.sum(y_values)
y_values /= total_sum

# 可视化序列
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o')
plt.title('Exponential Decay Sequence Visualization')
plt.xlabel('Index of x')
plt.ylabel('Exponential Value')
plt.grid(True)
plt.show()