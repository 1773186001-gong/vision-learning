
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SimpleBackbone(nn.Module):
    """简化骨干网络，提取四个尺度的特征"""
    def __init__(self):
        super().__init__()
        self.stem = ConvBlock(3, 32, stride=2)   # 224 -> 112
        self.stage1 = ConvBlock(32, 64, stride=2)   # 112 -> 56
        self.stage2 = ConvBlock(64, 128, stride=2)  # 56 -> 28
        self.stage3 = ConvBlock(128, 256, stride=2) # 28 -> 14
        self.stage4 = ConvBlock(256, 512, stride=2) # 14 -> 7

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c2, c3, c4, c5


class SimpleFPN(nn.Module):
    """简化 FPN：统一通道数，并进行自上而下的特征融合"""
    def __init__(self, out_channels=128):
        super().__init__()

        # 横向连接：把不同层的通道数统一
        self.lat2 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(512, out_channels, kernel_size=1)

        # 输出卷积：进一步处理融合后的特征
        self.out2 = ConvBlock(out_channels, out_channels)
        self.out3 = ConvBlock(out_channels, out_channels)
        self.out4 = ConvBlock(out_channels, out_channels)
        self.out5 = ConvBlock(out_channels, out_channels)

    def forward(self, features):
        c2, c3, c4, c5 = features

        # 先处理最深层特征
        p5 = self.lat5(c5)

        # 从深层向浅层逐级上采样并相加
        p4 = self.lat4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="nearest"
        )
        p3 = self.lat3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="nearest"
        )
        p2 = self.lat2(c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="nearest"
        )

        # 输出多尺度特征
        return (
            self.out2(p2),
            self.out3(p3),
            self.out4(p4),
            self.out5(p5)
        )


class SimpleDetectionHead(nn.Module):
    """每个尺度分别预测类别分数和边界框参数"""
    def __init__(self, in_channels=128, num_classes=2):
        super().__init__()

        self.class_heads = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
            for _ in range(4)
        ])

        self.box_heads = nn.ModuleList([
            nn.Conv2d(in_channels, 4, kernel_size=1)
            for _ in range(4)
        ])

    def forward(self, features):
        results = []

        for feature, cls_head, box_head in zip(
            features, self.class_heads, self.box_heads
        ):
            class_logits = cls_head(feature)
            box_predictions = box_head(feature)

            results.append({
                "class_logits": class_logits,
                "box_predictions": box_predictions
            })

        return results


class SimpleDetector(nn.Module):
    """Backbone + FPN Neck + Detection Head"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.neck = SimpleFPN(out_channels=128)
        self.head = SimpleDetectionHead(
            in_channels=128,
            num_classes=num_classes
        )

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        predictions = self.head(neck_features)

        return backbone_features, neck_features, predictions


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟一张 224x224 的 RGB 图片
    image = torch.randn(1, 3, 224, 224)

    model = SimpleDetector(num_classes=2)
    model.eval()

    with torch.no_grad():
        backbone_features, neck_features, predictions = model(image)

    print("========== Detection Architecture ==========")
    print(f"Input image shape: {tuple(image.shape)}")

    print("\n[Backbone features]")
    for name, feature in zip(
        ["C2", "C3", "C4", "C5"], backbone_features
    ):
        print(f"{name}: {tuple(feature.shape)}")

    print("\n[Neck / FPN features]")
    for name, feature in zip(
        ["P2", "P3", "P4", "P5"], neck_features
    ):
        print(f"{name}: {tuple(feature.shape)}")

    print("\n[Detection Head predictions]")
    for index, result in enumerate(predictions, start=2):
        print(
            f"P{index} class logits: "
            f"{tuple(result['class_logits'].shape)}"
        )
        print(
            f"P{index} box predictions: "
            f"{tuple(result['box_predictions'].shape)}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n[Model parameters]")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
