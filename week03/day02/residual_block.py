import torch
import torch.nn as nn


# =========================
# 1. 定义一个残差块
# =========================
class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(channels)

        # ReLU 激活函数
        self.relu = nn.ReLU()

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # Batch Normalization
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        # 保存原始输入
        identity = x

        # 第一层卷积
        out = self.conv1(x)

        # BN
        out = self.bn1(out)

        # ReLU
        out = self.relu(out)

        # 第二层卷积
        out = self.conv2(out)

        # BN
        out = self.bn2(out)

        # 残差连接
        out = out + identity

        # 最后的 ReLU
        out = self.relu(out)

        return out


# =========================
# 2. 创建一个残差块
# =========================
block = ResidualBlock(64)


# =========================
# 3. 创建一个假的输入
# =========================
x = torch.randn(
    4,      # batch size
    64,     # channels
    56,     # height
    56      # width
)


# =========================
# 4. 前向传播
# =========================
y = block(x)


# =========================
# 5. 查看输入输出
# =========================
print("输入 Tensor：")
print(x.shape)

print("\n输出 Tensor：")
print(y.shape)

print("\n输入元素数量：", x.numel())
print("输出元素数量：", y.numel())