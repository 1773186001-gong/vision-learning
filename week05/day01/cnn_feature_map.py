
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms


DATA_DIR = Path(r"D:\python\test\vision-learning\data\train")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root=str(DATA_DIR),
        transform=transform
    )

    print("类别：", dataset.classes)
    print("图片数量：", len(dataset))

    image, label = dataset[0]
    x = image.unsqueeze(0)

    print("\n输入形状：", tuple(x.shape))

    model = SimpleCNN()
    model.eval()

    # 逐层观察特征提取部分的输出形状
    with torch.no_grad():
        for index, layer in enumerate(model.features):
            x = layer(x)
            print(
                f"第 {index + 1:02d} 层 "
                f"{layer.__class__.__name__:12s} "
                f"输出形状：{tuple(x.shape)}"
            )

    print("\n卷积层参数量：")
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            count = sum(p.numel() for p in layer.parameters())
            print(
                f"{name}: "
                f"{layer.in_channels} -> {layer.out_channels}, "
                f"kernel={layer.kernel_size}, "
                f"参数量={count:,}"
            )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n整个模型参数量：{total_params:,}")


if __name__ == "__main__":
    main()
