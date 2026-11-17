
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ==================================================
# 1. 路径与参数
# ==================================================
PROJECT_DIR = Path(r"D:\python\test\vision-learning")
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

SAVE_DIR = PROJECT_DIR / "week05" / "day03"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "small_cnn_trained.pth"
MATRIX_PATH = SAVE_DIR / "small_cnn_confusion_matrix.png"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", DEVICE)
print("训练集路径：", TRAIN_DIR)
print("测试集路径：", TEST_DIR)


# ==================================================
# 2. 检查数据目录
# ==================================================
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"找不到训练集：{TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"找不到测试集：{TEST_DIR}")


# ==================================================
# 3. 图片预处理
# ==================================================
# 先使用简单、统一的预处理。
# 如果原训练代码使用了 Normalize，
# 后续比较实验时应统一两边的预处理。

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)



print("\n========== 数据增强检查 ==========")

image1, label1 = train_dataset[0]
image2, label2 = train_dataset[0]

difference = (image1 - image2).abs().mean().item()

print("图片类别：", train_dataset.classes[label1])
print("第一次图片形状：", tuple(image1.shape))
print("第二次图片形状：", tuple(image2.shape))
print("两次图片平均绝对差异：", difference)

if difference > 0:
    print("检查结果：图片像素发生变化，随机增强可能已生效。")
else:
    print("检查结果：两次图片相同；请检查 transform 配置。")

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=test_transform
)


print("\n训练集类别：", train_dataset.classes)
print("测试集类别：", test_dataset.classes)
print("训练集图片数量：", len(train_dataset))
print("测试集图片数量：", len(test_dataset))

if train_dataset.classes != test_dataset.classes:
    raise ValueError("训练集与测试集的类别顺序不一致！")

if len(train_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("训练集或测试集为空！")

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

class_names = train_dataset.classes
num_classes = len(class_names)


# ==================================================
# 4. 定义减少参数版 CNN
# ==================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # 输入：3 x 224 x 224
            nn.Conv2d(
                3, 32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 输出：32 x 112 x 112
            nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 输出：64 x 56 x 56
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # 输出：128 x 28 x 28
        )

        # 将每个通道的 28 x 28 特征图
        # 压缩为一个数
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 128 个通道对应 128 个输入特征
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


model = SmallCNN(num_classes=num_classes).to(DEVICE)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n模型总参数量：", f"{total_params:,}")
print("可训练参数量：", f"{trainable_params:,}")


# ==================================================
# 5. 训练模型
# ==================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

train_losses = []
train_accuracies = []

print("\n========== 开始训练 ==========")

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Acc: {epoch_accuracy:.2f}%"
    )


# ==================================================
# 6. 保存训练好的权重
# ==================================================
torch.save(model.state_dict(), MODEL_PATH)

print("\n模型权重已保存：", MODEL_PATH)


# ==================================================
# 7. 绘制训练曲线
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

plt.savefig(
    SAVE_DIR / "small_cnn_training_curves.png",
    dpi=150
)
plt.show()


# ==================================================
# 8. 在独立测试集上评估
# ==================================================
# 测试阶段不反向传播，不更新参数
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

test_correct = 0
test_total = 0

print("\n========== 开始测试 ==========")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        test_total += labels.size(0)
        test_correct += (
            predictions == labels
        ).sum().item()

        for true_label, pred_label in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        ):
            confusion[true_label, pred_label] += 1


test_accuracy = (
    100.0 * test_correct / test_total
    if test_total > 0 else 0.0
)

print("\n========== 测试结果 ==========")
print("测试图片总数：", test_total)
print("预测正确数量：", test_correct)
print(f"测试准确率：{test_accuracy:.2f}%")

print("\n混淆矩阵：")
print(confusion)
print("行表示真实类别，列表示预测类别。")


# ==================================================
# 9. 计算每个类别的精确率和召回率
# ==================================================
print("\n========== 分类指标 ==========")

for i, name in enumerate(class_names):
    true_count = int(confusion[i, :].sum())
    true_positive = int(confusion[i, i])
    predicted_count = int(confusion[:, i].sum())

    recall = (
        100.0 * true_positive / true_count
        if true_count > 0 else 0.0
    )

    precision = (
        100.0 * true_positive / predicted_count
        if predicted_count > 0 else 0.0
    )

    print(
        f"{name}: "
        f"正确识别 {true_positive}/{true_count}, "
        f"Precision: {precision:.2f}%, "
        f"Recall: {recall:.2f}%"
    )


# ==================================================
# 10. 绘制混淆矩阵
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
    f"Small CNN Confusion Matrix\n"
    f"Accuracy: {test_accuracy:.2f}%"
)

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j,
            i,
            str(confusion[i, j]),
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax)
fig.tight_layout()

plt.savefig(MATRIX_PATH, dpi=150)

print("\n混淆矩阵图片已保存：", MATRIX_PATH)

plt.show()

print("\n实验完成！")

# 检查模型在训练集上的预测情况
model.eval()

print("\n========== 训练集预测检查 ==========")

check_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=False
)

shown = 0

with torch.no_grad():
    for images, labels in check_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        for true_id, pred_id in zip(labels, predictions):
            true_name = class_names[true_id.item()]
            pred_name = class_names[pred_id.item()]

            print(
                f"真实类别：{true_name:<5} "
                f"预测类别：{pred_name}"
            )

            shown += 1
            if shown >= 10:
                break

        if shown >= 10:
            break
