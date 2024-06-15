import torch
from torch import nn

m = nn.Sigmoid()
loss = nn.BCELoss()


batch_size = 16
feature_size = 10
input = torch.randn(batch_size, feature_size, requires_grad=True)
target = torch.rand(batch_size, feature_size, requires_grad=False)

print(m(input))
print(target)

loss_ = loss(m(input), target)
loss_.backward()
print(loss_.item())
loss_2 = loss(m(input), m(input))
loss_2.backward()
print(loss_2.item())