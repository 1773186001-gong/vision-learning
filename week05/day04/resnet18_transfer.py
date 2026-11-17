
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt


# ==================================================
# 1. 路径与基本参数
# ==================================================
PROJECT_DIR = Path(r"D:\python\test\vision-learning")
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

SAVE_DIR = PROJECT_DIR / "week05" / "day04"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "resnet18_transfer.pth"
MATRIX_PATH = SAVE_DIR / "resnet18_confusion_matrix.png"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", DEVICE)


# ==================================================
# 2. 检查数据目录
# ==================================================
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"训练集不存在：{TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"测试集不存在：{TEST_DIR}")


# ==================================================
# 3. 数据预处理
# ==================================================
# 使用 ImageNet 预训练模型对应的归一化参数
weights = ResNet18_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=test_transform
)

class_names = train_dataset.classes

if class_names != test_dataset.classes:
    raise ValueError("训练集和测试集类别顺序不一致！")

print("类别：", class_names)
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==================================================
# 4. 加载预训练 ResNet18
# ==================================================
print("\n正在加载 ImageNet 预训练 ResNet18...")

model = models.resnet18(weights=weights)

# 冻结所有已有参数
for param in model.parameters():
    param.requires_grad = False

# 替换最后的分类层
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))

# 新分类层默认可训练
model = model.to(DEVICE)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("模型总参数量：", f"{total_params:,}")
print("可训练参数量：", f"{trainable_params:,}")
print("冻结特征层，只训练最后的 fc 分类层。")


# ==================================================
# 5. 训练分类层
# ==================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

train_losses = []
train_accuracies = []

print("\n========== 开始训练 ==========")

for epoch in range(EPOCHS):
    model.train()

    # 冻结卷积参数，但 BatchNorm 的运行统计量也会变化。
    # 为了保持预训练特征稳定，冻结特征层的 BatchNorm 行为。
    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Acc: {epoch_acc:.2f}%"
    )


# ==================================================
# 6. 保存模型
# ==================================================
torch.save(model.state_dict(), MODEL_PATH)

print("\n模型已保存：", MODEL_PATH)


# ==================================================
# 7. 测试集评估
# ==================================================
model.eval()

num_classes = len(class_names)

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

correct = 0
total = 0

print("\n========== 测试集评估 ==========")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for true_label, pred_label in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        ):
            confusion[true_label, pred_label] += 1

test_accuracy = (
    100.0 * correct / total if total else 0.0
)

print("测试图片总数：", total)
print("预测正确数量：", correct)
print(f"测试准确率：{test_accuracy:.2f}%")

print("\n混淆矩阵：")
print(confusion)
print("行是真实类别，列是预测类别。")


# ==================================================
# 8. 每类精确率与召回率
# ==================================================
print("\n========== 分类指标 ==========")

for i, name in enumerate(class_names):
    true_count = int(confusion[i, :].sum())
    true_positive = int(confusion[i, i])
    predicted_count = int(confusion[:, i].sum())

    recall = (
        100.0 * true_positive / true_count
        if true_count else 0.0
    )

    precision = (
        100.0 * true_positive / predicted_count
        if predicted_count else 0.0
    )

    print(
        f"{name}: "
        f"正确识别 {true_positive}/{true_count}, "
        f"Precision: {precision:.2f}%, "
        f"Recall: {recall:.2f}%"
    )


# ==================================================
# 9. 绘制混淆矩阵
# ==================================================
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(confusion)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(
    f"ResNet18 Transfer Learning\n"
    f"Test Accuracy: {test_accuracy:.2f}%"
)

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j, i, str(confusion[i, j]),
            ha="center", va="center"
        )

fig.colorbar(im, ax=ax)
fig.tight_layout()

plt.savefig(MATRIX_PATH, dpi=150)
print("\n混淆矩阵已保存：", MATRIX_PATH)

plt.show()


# ==================================================
# 10. 绘制训练曲线
# ==================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o"
)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(
    range(1, EPOCHS + 1),
    train_accuracies,
    marker="o"
)
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)
axes[1].grid(True)

fig.tight_layout()

curve_path = SAVE_DIR / "resnet18_training_curves.png"
plt.savefig(curve_path, dpi=150)

print("训练曲线已保存：", curve_path)

plt.show()

print("\n迁移学习实验完成！")
