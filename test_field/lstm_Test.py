
from torch import nn
import torch
rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.shape)