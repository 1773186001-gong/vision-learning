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

        self.bn1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        # 保存原始输入
        identity = x

        # 经过卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 没有 Shortcut 的结果
        without_shortcut = out

        # 加入 Shortcut
        with_shortcut = out + identity

        # 激活
        with_shortcut = self.relu(with_shortcut)

        return without_shortcut, with_shortcut


# 创建模型
block = ResidualBlock(64)

# 创建输入
x = torch.randn(4, 64, 56, 56)

# 前向传播
without_shortcut, with_shortcut = block(x)


print("输入：", x.shape)

print("\n没有 Shortcut：")
print(without_shortcut.shape)

print("\n加入 Shortcut：")
print(with_shortcut.shape)


# 比较数值
print("\n输入平均值：", x.mean().item())
print("没有 Shortcut 平均值：", without_shortcut.mean().item())
print("加入 Shortcut 平均值：", with_shortcut.mean().item())