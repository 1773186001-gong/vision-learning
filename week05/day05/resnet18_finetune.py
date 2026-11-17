
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader


# =========================
# 1. 基础设置
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

BASE_DIR = r"D:\python\test\vision-learning"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "week05", "day05")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 5

# 两组参数使用不同学习率：
# Layer4 是预训练层，小幅调整；
# fc 是新建的分类层，需要更积极地学习。
LR_LAYER4 = 0.0001
LR_FC = 0.001


# =========================
# 2. 数据预处理
# =========================
weights = ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_dataset.classes

print("类别：", class_names)
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))


# =========================
# 3. 加载预训练 ResNet18
# =========================
print("\n正在加载 ImageNet 预训练 ResNet18...")

model = models.resnet18(weights=weights)

# 先冻结全部参数
for param in model.parameters():
    param.requires_grad = False

# 解冻 Layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# 替换分类层，并允许它训练
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(class_names))

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"模型总参数量：{total_params:,}")
print(f"可训练参数量：{trainable_params:,}")
print("训练范围：Layer4 + fc")


# =========================
# 4. 损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": LR_LAYER4
    },
    {
        "params": model.fc.parameters(),
        "lr": LR_FC
    }
])

train_losses = []
train_accuracies = []


# =========================
# 5. 开始训练
# =========================
print("\n========== 开始微调 ==========")

for epoch in range(EPOCHS):
    model.train()

    # 小数据集下先固定 BatchNorm 的运行统计量，
    # 避免每个批次样本太少导致统计量波动过大。
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Acc: {epoch_acc:.2f}%"
    )


# =========================
# 6. 保存模型
# =========================
model_path = os.path.join(
    OUTPUT_DIR, "resnet18_finetune.pth"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, model_path)

print("\n模型已保存：", model_path)


# =========================
# 7. 测试集评估
# =========================
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

accuracy = (
    all_predictions == all_labels
).mean() * 100

print("\n========== 测试集评估 ==========")
print("测试图片总数：", len(all_labels))
print("预测正确数量：", int(
    (all_predictions == all_labels).sum()
))
print(f"测试准确率：{accuracy:.2f}%")


# =========================
# 8. 混淆矩阵和分类指标
# =========================
num_classes = len(class_names)
confusion = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for true_label, pred_label in zip(
    all_labels, all_predictions
):
    confusion[true_label, pred_label] += 1

print("\n混淆矩阵：")
print(confusion)
print("行是真实类别，列是预测类别。")

print("\n========== 分类指标 ==========")

for i, name in enumerate(class_names):
    tp = confusion[i, i]
    actual_count = confusion[i, :].sum()
    predicted_count = confusion[:, i].sum()

    recall = (
        tp / actual_count * 100
        if actual_count > 0 else 0
    )

    precision = (
        tp / predicted_count * 100
        if predicted_count > 0 else 0
    )

    print(
        f"{name}: 正确识别 {tp}/{actual_count}, "
        f"Precision: {precision:.2f}%, "
        f"Recall: {recall:.2f}%"
    )


# =========================
# 9. 保存混淆矩阵图片
# =========================
plt.figure(figsize=(6, 5))
plt.imshow(confusion, interpolation="nearest")
plt.title("ResNet18 Fine-tuning Confusion Matrix")
plt.colorbar()

ticks = np.arange(num_classes)
plt.xticks(ticks, class_names)
plt.yticks(ticks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j, i, str(confusion[i, j]),
            ha="center", va="center"
        )

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR, "resnet18_finetune_confusion_matrix.png"
)
plt.savefig(confusion_path, dpi=150)
plt.close()


# =========================
# 10. 保存训练曲线
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
axes[1].grid(True)

plt.tight_layout()

curve_path = os.path.join(
    OUTPUT_DIR, "resnet18_finetune_training_curves.png"
)
plt.savefig(curve_path, dpi=150)
plt.close()

print("\n混淆矩阵已保存：", confusion_path)
print("训练曲线已保存：", curve_path)
print("\nResNet18 微调实验完成！")
