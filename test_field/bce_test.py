import torch
import torch.nn as nn

# 定义二分类的BCELoss函数
loss_fn = nn.BCELoss(weight=torch.tensor(2.0))

# 假设我们的预测值和目标值
logits = torch.tensor([0.8, 0.3, 0.6])
targets = torch.tensor([1, 0, 1])


print(logits.shape)
print(targets.shape)

# 计算损失
loss = loss_fn(logits, targets.float())

print(loss)

