import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================
# 1. 设置设备
# =========================
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
# 3. 加载数据集
# =========================
train_dir = r"D:\python\test\vision-learning\data\train"
test_dir = r"D:\python\test\vision-learning\data\test"

train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=transform
)

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

print("类别：", train_dataset.classes)
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))


# =========================
# 4. 创建 ResNet18
# =========================
model = models.resnet18(weights=None)

# 修改最后的全连接层
num_classes = len(train_dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

print("\n模型创建完成")
print(model)


# =========================
# 5. 损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 6. 训练函数
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算 Loss
        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# =========================
# 7. 测试函数
# =========================
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# =========================
# 8. 开始训练
# =========================
epochs = 3

print("\n========== 开始训练 ==========")

for epoch in range(epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_acc:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Accuracy: {test_acc:.2f}%"
    )


print("\n========== 训练结束 ==========")


# =========================
# 9. 保存模型
# =========================
torch.save(
    model.state_dict(),
    "resnet18.pth"
)

print("模型已经保存为：resnet18.pth")