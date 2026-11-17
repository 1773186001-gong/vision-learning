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
# 3. 训练集数据增强
# ========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================================
# 4. 测试集不做随机增强
# ========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================================
# 5. 创建数据集
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
# 6. 创建 DataLoader
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
# 7. 创建 ResNet18
# ========================================

model = models.resnet18(weights=None)

# 修改最后的全连接层
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)


# ========================================
# 8. 损失函数和优化器
# ========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ========================================
# 9. 训练函数
# ========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    average_loss = running_loss / len(train_loader)

    return average_loss, accuracy


# ========================================
# 10. 测试函数
# ========================================

def evaluate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    average_loss = running_loss / len(test_loader)

    return average_loss, accuracy


# ========================================
# 11. 开始训练
# ========================================

epochs = 5

for epoch in range(epochs):

    train_loss, train_accuracy = train_one_epoch()

    test_loss, test_accuracy = evaluate()

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# ========================================
# 12. 保存模型
# ========================================

torch.save(
    model.state_dict(),
    "resnet18_aug.pth"
)

print("模型已经保存为：resnet18_aug.pth")