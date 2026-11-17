import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # 主分支
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # 残差连接
        out = out + x

        # 激活
        out = self.relu(out)

        return out


# 创建一个 Residual Block
model = ResidualBlock(16)

# 模拟一批图片特征
x = torch.randn(4, 16, 32, 32)

# 前向传播
y = model(x)

print("输入形状：", x.shape)
print("输出形状：", y.shape)