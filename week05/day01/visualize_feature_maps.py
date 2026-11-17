from torch.utils.data import DataLoader
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 1. 数据路径
DATA_DIR = r"D:\python\test\vision-learning\data\train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("类别：", dataset.classes)
print("图片数量：", len(dataset))

# 2. 定义与之前实验一致的 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

import torch

# 根据实际保存位置修改这两个路径
random_path = "cnn_random.pth"
trained_path = "cnn_trained.pth"

random_state = torch.load(
    random_path, map_location="cpu", weights_only=True
)
trained_state = torch.load(
    trained_path, map_location="cpu", weights_only=True
)

# 第一层卷积的权重，形状为 [32, 3, 3, 3]
w_before = random_state["features.0.weight"]
w_after = trained_state["features.0.weight"]

print("训练前权重形状：", tuple(w_before.shape))
print("训练后权重形状：", tuple(w_after.shape))

# 计算参数变化程度
diff = (w_after - w_before).abs()

print("平均绝对变化：", diff.mean().item())
print("最大绝对变化：", diff.max().item())
print("发生变化的参数比例：", (diff > 1e-8).float().mean().item())



# 3. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 保存随机初始化的模型参数
torch.save(model.state_dict(), "cnn_random.pth")

# 4. 选择同一张图片，记录训练前的特征
image, label = dataset[0]
input_tensor = image.unsqueeze(0).to(device)

def extract_features(model, x):
    model.eval()
    with torch.no_grad():
        # 第一组卷积、BN、ReLU之后
        first = model.features[:3](x)

        # 第三组卷积、BN、ReLU之后
        third = model.features[:11](x)

    return first[0].cpu(), third[0].cpu()

first_before, third_before = extract_features(model, input_tensor)

# 5. 在现有40张图片上训练5轮，仅用于观察特征变化
loader = DataLoader(dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch [{epoch + 1}/5] "
        f"Loss: {total_loss / total:.4f} "
        f"Train Acc: {correct / total:.2%}"
    )

# 保存训练后的模型
torch.save(model.state_dict(), "cnn_trained.pth")

# ===== Week05 Day02：卷积核权重对比 =====

# 加载训练前后的权重
random_state = torch.load(
    "cnn_random.pth",
    map_location="cpu",
    weights_only=True
)

trained_state = torch.load(
    "cnn_trained.pth",
    map_location="cpu",
    weights_only=True
)

# 取第一层卷积的权重
# 形状：[32, 3, 3, 3]
w_before = random_state["features.0.weight"]
w_after = trained_state["features.0.weight"]

print("\n===== 第一层卷积核权重对比 =====")
print("训练前权重形状：", tuple(w_before.shape))
print("训练后权重形状：", tuple(w_after.shape))

# 计算权重变化
diff = (w_after - w_before).abs()

print(f"平均绝对变化：{diff.mean().item():.8f}")
print(f"最大绝对变化：{diff.max().item():.8f}")
print(
    "发生变化的参数比例："
    f"{(diff > 1e-8).float().mean().item():.2%}"
)

# ===== 可视化前8个卷积核 =====
# 每个卷积核有RGB三个通道。
# 这里对RGB权重取平均，方便显示空间权重分布。
before_kernels = w_before[:8].mean(dim=1).numpy()
after_kernels = w_after[:8].mean(dim=1).numpy()

# 统一颜色范围，方便比较
limit = max(
    abs(before_kernels).max(),
    abs(after_kernels).max(),
    abs(before_kernels).min(),
    abs(after_kernels).min()
)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))

for i in range(8):
    axes[0, i].imshow(
        before_kernels[i],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit
    )
    axes[0, i].set_title(f"Before {i}")
    axes[0, i].axis("off")

    axes[1, i].imshow(
        after_kernels[i],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit
    )
    axes[1, i].set_title(f"After {i}")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Before Training")
axes[1, 0].set_ylabel("After Training")

plt.suptitle("First Layer Convolution Kernels")
plt.tight_layout()
plt.show()

# 6. 用相同图片提取训练后的特征
first_after, third_after = extract_features(model, input_tensor)

print("训练前第一层形状：", tuple(first_before.shape))
print("训练后第一层形状：", tuple(first_after.shape))
print("训练前第三层形状：", tuple(third_before.shape))
print("训练后第三层形状：", tuple(third_after.shape))

# 7. 对比第一组卷积特征图
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

axes[0, 0].imshow(image.permute(1, 2, 0))
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for i in range(4):
    axes[0, i + 1].imshow(first_before[i], cmap="viridis")
    axes[0, i + 1].set_title(f"Before Ch {i}")
    axes[0, i + 1].axis("off")

    axes[1, i + 1].imshow(first_after[i], cmap="viridis")
    axes[1, i + 1].set_title(f"After Ch {i}")
    axes[1, i + 1].axis("off")

axes[1, 0].text(
    0.5, 0.5, "After Training",
    ha="center", va="center"
)
axes[1, 0].axis("off")

plt.tight_layout()
plt.show()

# 8. 对比第三组卷积的前8个通道
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i in range(8):
    axes[i // 4, i % 4].imshow(
        third_before[i], cmap="viridis"
    )
    axes[i // 4, i % 4].set_title(f"Before Ch {i}")
    axes[i // 4, i % 4].axis("off")

    axes[i // 4 + 2, i % 4].imshow(
        third_after[i], cmap="viridis"
    )
    axes[i // 4 + 2, i % 4].set_title(f"After Ch {i}")
    axes[i // 4 + 2, i % 4].axis("off")

plt.tight_layout()
plt.show()
