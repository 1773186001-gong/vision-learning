
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePAN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()

        # 第一步：统一各层的通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 第二步：FPN 自上而下融合后的特征处理
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            )
            for _ in in_channels_list
        ])

        # 第三步：PAN 自下而上的下采样
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=2, padding=1
            )
            for _ in range(len(in_channels_list) - 1)
        ])

        # 第四步：处理 PAN 融合后的特征
        self.pan_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            )
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, features):
        # features 顺序：C2、C3、C4、C5，由浅到深
        laterals = [
            conv(feature)
            for conv, feature in zip(
                self.lateral_convs, features
            )
        ]

        # -------------------------
        # A. FPN：自上而下融合
        # -------------------------
        top_down = [None] * len(laterals)
        top_down[-1] = laterals[-1]

        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                top_down[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest"
            )

            top_down[i] = laterals[i] + upsampled

        fpn_features = [
            conv(feature)
            for conv, feature in zip(
                self.fpn_convs, top_down
            )
        ]

        # -------------------------
        # B. PAN：自下而上融合
        # -------------------------
        pan_features = [None] * len(fpn_features)
        pan_features[0] = fpn_features[0]

        for i in range(len(fpn_features) - 1):
            # 将当前浅层特征下采样
            downsampled = self.downsample_convs[i](
                pan_features[i]
            )

            # 与对应的 FPN 特征融合
            # 两者的通道数和空间尺寸应一致
            fused = downsampled + fpn_features[i + 1]

            pan_features[i + 1] = self.pan_convs[i](fused)

        return fpn_features, pan_features


def main():
    torch.manual_seed(42)

    # 模拟骨干网络的四层输出
    features = [
        torch.randn(1, 64, 56, 56),    # C2
        torch.randn(1, 128, 28, 28),   # C3
        torch.randn(1, 256, 14, 14),   # C4
        torch.randn(1, 512, 7, 7)      # C5
    ]

    print("输入特征：")
    for name, feature in zip(
        ["C2", "C3", "C4", "C5"], features
    ):
        print(f"{name}: {list(feature.shape)}")

    model = SimplePAN(
        in_channels_list=[64, 128, 256, 512],
        out_channels=128
    )

    model.eval()
    with torch.no_grad():
        fpn_outputs, pan_outputs = model(features)

    print("\nFPN 输出（自上而下融合）：")
    for name, output in zip(
        ["P2", "P3", "P4", "P5"], fpn_outputs
    ):
        print(f"{name}: {list(output.shape)}")

    print("\nPAN 输出（进一步自下而上融合）：")
    for name, output in zip(
        ["N2", "N3", "N4", "N5"], pan_outputs
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
    print("\nPAN 前向传播完成！")


if __name__ == "__main__":
    main()
