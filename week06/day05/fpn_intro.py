
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()

        # 1. 使用 1×1 卷积，将不同层的通道数统一
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # 2. 使用 3×3 卷积，对融合后的特征进一步处理
        self.output_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
            for _ in in_channels_list
        ])

    def forward(self, features):
        # features 按照从浅层到深层的顺序传入
        # 例如 C2、C3、C4、C5
        laterals = [
            conv(feature)
            for conv, feature in zip(
                self.lateral_convs, features
            )
        ]

        # 最深层先作为金字塔的起点
        fused = [None] * len(laterals)
        fused[-1] = laterals[-1]

        # 从深层向浅层逐级融合
        for i in range(len(laterals) - 2, -1, -1):
            # 将更深一层的特征放大到当前层的尺寸
            upsampled = F.interpolate(
                fused[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest"
            )

            # 融合：深层上采样特征 + 当前层横向特征
            fused[i] = laterals[i] + upsampled

        # 使用 3×3 卷积生成最终输出
        outputs = [
            conv(feature)
            for conv, feature in zip(
                self.output_convs, fused
            )
        ]

        return outputs


def main():
    torch.manual_seed(42)

    # 模拟骨干网络提取的四层特征
    # 实际项目中，这些特征来自 CNN 或 ResNet
    features = [
        torch.randn(1, 64, 56, 56),   # C2
        torch.randn(1, 128, 28, 28),  # C3
        torch.randn(1, 256, 14, 14),  # C4
        torch.randn(1, 512, 7, 7)     # C5
    ]

    print("输入特征：")
    for name, feature in zip(
        ["C2", "C3", "C4", "C5"], features
    ):
        print(f"{name}: {list(feature.shape)}")

    model = SimpleFPN(
        in_channels_list=[64, 128, 256, 512],
        out_channels=128
    )

    # 当前只检查前向传播，不进行训练
    model.eval()
    with torch.no_grad():
        outputs = model(features)

    print("\nFPN 输出特征：")
    for name, output in zip(
        ["P2", "P3", "P4", "P5"], outputs
    ):
        print(f"{name}: {list(output.shape)}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    print("\nFPN 前向传播完成！")


if __name__ == "__main__":
    main()
