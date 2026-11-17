
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """简化版通道注意力"""

    def __init__(self, channels):
        super().__init__()

        # 将每个通道的空间信息压缩成一个数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 学习不同通道的权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # [B, C, H, W] -> [B, C, 1, 1]
        pooled = self.avg_pool(x)

        # [B, C, 1, 1] -> [B, C]
        pooled = pooled.view(b, c)

        # [B, C] -> [B, C, 1, 1]
        weights = self.fc(pooled).view(b, c, 1, 1)

        # 每个通道乘上对应权重
        out = x * weights

        return out, weights


class SpatialAttention(nn.Module):
    """简化版空间注意力"""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度分别计算平均值和最大值
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]

        # [B, 1, H, W] + [B, 1, H, W]
        # 拼接为 [B, 2, H, W]
        combined = torch.cat([avg_map, max_map], dim=1)

        # 生成空间权重 [B, 1, H, W]
        weights = self.sigmoid(self.conv(combined))

        # 每个空间位置乘上对应权重
        out = x * weights

        return out, weights


def main():
    torch.manual_seed(42)

    # 模拟 ResNet 中间层输出的特征图
    x = torch.randn(1, 64, 56, 56)

    print("原始特征图形状:", x.shape)

    # 通道注意力
    channel_attention = ChannelAttention(channels=64)
    channel_out, channel_weights = channel_attention(x)

    print("\n--- 通道注意力 ---")
    print("输出特征图:", channel_out.shape)
    print("通道权重形状:", channel_weights.shape)
    print("权重最小值:", channel_weights.min().item())
    print("权重最大值:", channel_weights.max().item())
    print("前10个通道权重:",
          channel_weights[0, :10, 0, 0].detach().tolist())

    # 空间注意力
    spatial_attention = SpatialAttention()
    spatial_out, spatial_weights = spatial_attention(x)

    print("\n--- 空间注意力 ---")
    print("输出特征图:", spatial_out.shape)
    print("空间权重形状:", spatial_weights.shape)
    print("权重最小值:", spatial_weights.min().item())
    print("权重最大值:", spatial_weights.max().item())

    # 比较输入和输出的平均绝对值
    print("\n--- 特征变化 ---")
    print("输入平均绝对值:", x.abs().mean().item())
    print("通道注意力后:",
          channel_out.abs().mean().item())
    print("空间注意力后:",
          spatial_out.abs().mean().item())


if __name__ == "__main__":
    main()
