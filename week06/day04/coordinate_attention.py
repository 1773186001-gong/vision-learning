
import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention（坐标注意力）

    输入: [B, C, H, W]
    输出: [B, C, H, W]
    """

    def __init__(self, channels, reduction=32):
        super().__init__()

        # 隐藏通道数，避免出现 0
        hidden_channels = max(8, channels // reduction)

        # 沿宽度方向池化，保留高度
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        # 沿高度方向池化，保留宽度
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 两个方向共享的特征变换层
        self.conv1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.Hardswish()

        # 分别生成高度和宽度方向的注意力权重
        self.conv_h = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv_w = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        batch_size, channels, height, width = x.shape

        # 1. 两个方向分别池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        # x_h: [B, C, H, 1]
        # x_w: [B, C, 1, W]

        # 2. 将宽度方向的结果转置为 [B, C, W, 1]
        x_w = x_w.permute(0, 1, 3, 2)

        # 3. 沿空间序列维度拼接
        y = torch.cat([x_h, x_w], dim=2)

        # [B, C, H + W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 4. 根据原始 H 和 W 拆分两个方向
        x_h, x_w = torch.split(
            y,
            [height, width],
            dim=2
        )

        # 恢复宽度分支的布局
        x_w = x_w.permute(0, 1, 3, 2)

        # 5. 分别生成两个方向的注意力权重
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        # 6. 两个方向的权重共同作用于原始特征
        out = identity * a_h * a_w

        return out, a_h, a_w


def main():
    torch.manual_seed(42)

    # 模拟 CNN 中间层输出的特征图
    x = torch.randn(1, 64, 56, 56)

    # 创建 CA 模块
    ca = CoordinateAttention(channels=64, reduction=32)

    print("========== 输入特征 ==========")
    print("输入形状:", x.shape)
    print("输入平均绝对值:", x.abs().mean().item())

    # 本次只测试前向传播，不进行训练
    ca.eval()

    with torch.no_grad():
        out, a_h, a_w = ca(x)

    print("\n========== 坐标注意力输出 ==========")
    print("最终输出形状:", out.shape)
    print("高度注意力权重形状:", a_h.shape)
    print("宽度注意力权重形状:", a_w.shape)

    print("\n========== 高度注意力权重 ==========")
    print("最小值:", a_h.min().item())
    print("最大值:", a_h.max().item())
    print("前 10 个值:")
    print(a_h[0, 0, :10, 0].tolist())

    print("\n========== 宽度注意力权重 ==========")
    print("最小值:", a_w.min().item())
    print("最大值:", a_w.max().item())
    print("前 10 个值:")
    print(a_w[0, 0, 0, :10].tolist())

    print("\n========== 特征幅度对比 ==========")
    print("输入平均绝对值:", x.abs().mean().item())
    print("输出平均绝对值:", out.abs().mean().item())

    total_params = sum(p.numel() for p in ca.parameters())
    trainable_params = sum(
        p.numel() for p in ca.parameters()
        if p.requires_grad
    )

    print("\n========== 参数统计 ==========")
    print("CA 模块总参数量:", total_params)
    print("CA 模块可训练参数量:", trainable_params)


if __name__ == "__main__":
    main()
