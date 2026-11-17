import torch
import torch.nn as nn


# 假设模型输出 3 个类别的分数
prediction = torch.tensor([
    [2.0, 1.0, 0.1]
])

# 真实类别是第 0 类
label = torch.tensor([0])


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 计算 Loss
loss = loss_fn(prediction, label)

print("Loss:", loss)