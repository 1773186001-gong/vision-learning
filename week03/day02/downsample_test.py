import torch
import torch.nn as nn


# 创建两个尺寸不同的 Tensor
x = torch.randn(4, 64, 56, 56)

out = torch.randn(4, 128, 28, 28)

print("x 的尺寸：", x.shape)
print("out 的尺寸：", out.shape)

print("\n尝试进行残差相加：")

result = out + x

print(result.shape)