import torch
from torch import nn

rnn = nn.LSTM(1,2,1, batch_first=True)
input = torch.randn(3, 4, 1)
output, _ = rnn(input)

print('input', input)
print('output', output)

# ex = torch.tensor([[[0.1204, 0.1738],
#          [0.1169, 0.2415],
#          [0.1324, 0.3535],
#          [0.0970, 0.3110]],
#
#         [[0.0665, 0.0714],
#          [0.1004, 0.1558],
#          [0.1259, 0.2704],
#          [0.0998, 0.2592]],
#
#         [[0.1124, 0.1581],
#          [0.1019, 0.1881],
#          [0.1074, 0.2464],
#          [0.1247, 0.3426]]])
#
# print(ex[:,-1,:])
