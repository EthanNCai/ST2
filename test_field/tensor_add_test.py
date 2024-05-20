import torch


numbers = [1., 2., 3., 4., 5., 6., 7., 8., 9.]

tensor = torch.tensor(numbers)

print(tensor)

tensor = tensor +tensor

print(tensor)

tensor =tensor *0.1

print(tensor)