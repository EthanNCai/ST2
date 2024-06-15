import numpy as np
import torch

data = np.array([[[1.1, 1.2], [2.1, 2.2], [3.1, 3.2], [4.1, 4.2]], [[5.1, 5.2], [6.1, 6.2], [7.1, 7.2], [8.1, 8.2]]])
print(data.shape)
data_tensor = torch.tensor(data)
print(data_tensor)
print(data_tensor.shape)

data_tensor = data_tensor.unsqueeze(-1)
print(data_tensor)
print(data_tensor.shape)
