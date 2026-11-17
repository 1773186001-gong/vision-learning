import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ========================================
# 1. 设置设备
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备：", device)


# ========================================
# 2. 数据路径
# ========================================

train_dir = r"D:\python\test\vision-learning\data\train"
test_dir = r"D:\python\test\vision-learning\data\test"


# ========================================
# 3. 数据预处理
# ========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================================
# 4. 创建数据集
# ========================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)


print("类别：", train_dataset.classes)

print("训练集数量：", len(train_dataset))

print("测试集数量：", len(test_dataset))


# ========================================
# 5. DataLoader
# ========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# ========================================
# 6. 加载预训练 ResNet18
# ========================================

model = models.resnet18(weights="DEFAULT")


# ========================================
# 7. 冻结所有参数
# ========================================

for param in model.parameters():
    param.requires_grad = False


# ========================================
# 8. 修改最后分类层
# ========================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(device)


# ========================================
# 9. 损失函数
# ========================================

criterion = nn.CrossEntropyLoss()


# ========================================
# 10. 只优化最后的 FC 层
# ========================================

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# ========================================
# 11. 训练
# ========================================

epochs = 5

for epoch in range(epochs):

    model.train()

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

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    train_accuracy = 100 * correct / total


    # ====================================
    # 测试
    # ====================================

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)

            test_correct += (predicted == labels).sum().item()


    test_accuracy = 100 * test_correct / test_total


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# ========================================
# 12. 保存模型
# ========================================

torch.save(
    model.state_dict(),
    "resnet18_transfer.pth"
)

print("模型已经保存为：resnet18_transfer.pth")