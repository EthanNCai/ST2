# target output size of 5
from torch import nn
import torch

m = nn.AdaptiveMaxPool1d(1)

# [batch, channel, len]
input = torch.randn(32, 1, 12)
output = m(input)



print(input.shape)
print(output.shape)


input2 = torch.randn(32, 1, 142)
output2 = m(input2)
print(input2.shape)
print(output2.shape)