
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        # 平均池化和最大池化分别提取通道统计信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两种池化共用同一个 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        avg = self.avg_pool(x).view(batch_size, channels)
        max_value = self.max_pool(x).view(batch_size, channels)

        # 两种信息经过共享 MLP 后相加
        avg_out = self.mlp(avg)
        max_out = self.mlp(max_value)

        weights = self.sigmoid(avg_out + max_out)

        # [B, C] -> [B, C, 1, 1]
        weights = weights.view(batch_size, channels, 1, 1)

        # 广播乘法：每个通道使用自己的权重
        out = x * weights

        return out, weights


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size 应为 3 或 7")

        padding = kernel_size // 2

        # 输入是两张空间图，输出是一张空间权重图
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度 C 上进行平均池化和最大池化
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.max(x, dim=1, keepdim=True)[0]

        # 拼接后形状：[B, 2, H, W]
        combined = torch.cat([avg, max_value], dim=1)

        # 卷积后形状：[B, 1, H, W]
        weights = self.sigmoid(self.conv(combined))

        # 每个空间位置的权重应用于所有通道
        out = x * weights

        return out, weights


class CBAM(nn.Module):
    """完整 CBAM：先通道注意力，再空间注意力"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):
        channel_out, channel_weights = self.channel_attention(x)
        spatial_out, spatial_weights = self.spatial_attention(channel_out)

        return spatial_out, channel_weights, spatial_weights


def main():
    torch.manual_seed(42)

    # 模拟一个 CNN 输出的特征图
    x = torch.randn(1, 64, 56, 56)

    cbam = CBAM(channels=64, reduction=16, kernel_size=7)

    print("========== 输入 ==========")
    print("输入形状:", x.shape)
    print("输入平均绝对值:", x.abs().mean().item())

    # 本次只检查前向计算，不进行训练
    with torch.no_grad():
        out, channel_weights, spatial_weights = cbam(x)

    print("\n========== CBAM 输出 ==========")
    print("最终输出形状:", out.shape)
    print("通道权重形状:", channel_weights.shape)
    print("空间权重形状:", spatial_weights.shape)

    print("\n========== 通道注意力权重 ==========")
    print("最小值:", channel_weights.min().item())
    print("最大值:", channel_weights.max().item())
    print("前 10 个通道权重:")
    print(channel_weights[0, :10, 0, 0].tolist())

    print("\n========== 空间注意力权重 ==========")
    print("最小值:", spatial_weights.min().item())
    print("最大值:", spatial_weights.max().item())

    print("\n========== 特征幅度对比 ==========")
    print("输入平均绝对值:", x.abs().mean().item())
    print("输出平均绝对值:", out.abs().mean().item())

    total_params = sum(p.numel() for p in cbam.parameters())
    trainable_params = sum(
        p.numel() for p in cbam.parameters()
        if p.requires_grad
    )

    print("\n========== 参数统计 ==========")
    print("CBAM 总参数量:", total_params)
    print("CBAM 可训练参数量:", trainable_params)


if __name__ == "__main__":
    main()
