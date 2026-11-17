import torch
import torch.nn as nn

x = torch.rand(1, 3, 224, 224)

print("输入:", x.shape)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)

relu = nn.ReLU()

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

x = conv(x)
print("Conv:", x.shape)

x = relu(x)
print("ReLU:", x.shape)

x = pool(x)
print("Pool:", x.shape)