import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt


# =========================
# 1. 基础配置
# =========================

DATA_DIR = r"D:\python\test\vision-learning\data"
MODEL_PATH = r"D:\python\test\vision-learning\week03\day04\resnet18_transfer.pth"

BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备：", device)


# =========================
# 2. 数据预处理
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. 加载测试集
# =========================

test_dataset = datasets.ImageFolder(
    root=DATA_DIR + r"\test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("类别：", test_dataset.classes)
print("测试集数量：", len(test_dataset))


# =========================
# 4. 创建 ResNet18
# =========================

model = models.resnet18(weights=None)

# 猫狗二分类，所以最后输出 2 个类别
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# 加载 Week 3 Day 04 训练好的模型
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)

# 进入测试模式
model.eval()


# =========================
# 5. 开始预测
# =========================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # 找到分数最大的类别
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# =========================
# 6. 计算评价指标
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary"
)


print("\n==========================")
print("模型评价结果")
print("==========================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# =========================
# 7. 混淆矩阵
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n==========================")
print("Confusion Matrix")
print("==========================")

print(cm)


# =========================
# 8. 分类报告
# =========================

print("\n==========================")
print("Classification Report")
print("==========================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)


# =========================
# 9. 绘制混淆矩阵
# =========================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    test_dataset.classes
)

plt.yticks(
    [0, 1],
    test_dataset.classes
)

# 在矩阵中显示数字
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()