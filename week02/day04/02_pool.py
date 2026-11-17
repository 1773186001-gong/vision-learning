import torch
import torch.nn as nn

x = torch.rand(1, 16, 224, 224)

print("输入:", x.shape)

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

y = pool(x)

print("输出:", y.shape)
