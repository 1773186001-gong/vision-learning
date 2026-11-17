import torch
import torch.nn as nn

# 模拟一张 RGB 图片
x = torch.rand(1, 3, 224, 224)

print("输入:", x.shape)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3
)

y = conv(x)

print("输出:", y.shape)