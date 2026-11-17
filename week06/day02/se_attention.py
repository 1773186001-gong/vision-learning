
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块
    输入:  [B, C, H, W]
    输出:  [B, C, H, W]
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        # 第一步：Squeeze，全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 防止隐藏层维度变成 0
        hidden_channels = max(channels // reduction, 1)

        # 第二步：Excitation，学习每个通道的权重
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # [B, C, 1, 1] -> [B, C]
        y = y.view(batch_size, channels)

        # [B, C] -> [B, C]
        weights = self.fc(y)

        # [B, C] -> [B, C, 1, 1]
        weights = weights.view(batch_size, channels, 1, 1)

        # 第三步：Scale，按通道加权
        out = x * weights

        return out, weights


def main():
    # 为了让本次演示结果可以复现
    torch.manual_seed(42)

    # 模拟 CNN 输出的特征图
    x = torch.randn(1, 64, 56, 56)

    print("========== 输入特征 ==========")
    print("输入形状:", x.shape)
    print("输入平均绝对值:", x.abs().mean().item())

    # 创建 SE 模块
    se = SEBlock(channels=64, reduction=16)

    # 执行前向传播
    with torch.no_grad():
        out, weights = se(x)

    print("\n========== SE 模块 ==========")
    print("输出特征形状:", out.shape)
    print("通道权重形状:", weights.shape)
    print("隐藏层通道数:", max(64 // 16, 1))

    print("\n========== 通道权重 ==========")
    print("权重最小值:", weights.min().item())
    print("权重最大值:", weights.max().item())
    print("前 10 个通道权重:")
    print(weights[0, :10, 0, 0].tolist())

    print("\n========== 加权前后对比 ==========")
    print("输入平均绝对值:", x.abs().mean().item())
    print("SE 输出平均绝对值:", out.abs().mean().item())

    # 统计 SE 模块自身的可训练参数
    total_params = sum(p.numel() for p in se.parameters())
    trainable_params = sum(
        p.numel() for p in se.parameters() if p.requires_grad
    )

    print("\n========== 参数统计 ==========")
    print("SE 模块总参数量:", total_params)
    print("SE 模块可训练参数量:", trainable_params)


if __name__ == "__main__":
    main()
