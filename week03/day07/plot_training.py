import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# =========================
# 1. 基本配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

train_dir = r"D:\python\test\vision-learning\data\train"
test_dir = r"D:\python\test\vision-learning\data\test"


# =========================
# 2. 数据预处理
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

print("类别：", train_dataset.classes)
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))


# =========================
# 3. 加载预训练 ResNet18
# =========================

model = models.resnet18(weights="DEFAULT")

# 冻结全部参数
for param in model.parameters():
    param.requires_grad = False

# 解冻 Layer3、Layer4
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# 修改最后分类层
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)


# =========================
# 4. 损失函数和优化器
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)


# =========================
# 5. 记录训练数据
# =========================

num_epochs = 5

train_losses = []
train_accuracies = []
test_accuracies = []


# =========================
# 6. 开始训练
# =========================

for epoch in range(num_epochs):

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

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # =========================
    # 测试
    # =========================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    # 保存结果
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# =========================
# 7. 保存模型
# =========================

torch.save(
    model.state_dict(),
    "resnet18_day07.pth"
)

print("模型已保存：resnet18_day07.pth")


# =========================
# 8. 绘制 Loss 曲线
# =========================

epochs = range(1, num_epochs + 1)

plt.figure()

plt.plot(
    epochs,
    train_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.xticks(list(epochs))

plt.grid()

plt.show()


# =========================
# 9. 绘制 Accuracy 曲线
# =========================

plt.figure()

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    test_accuracies,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")

plt.xticks(list(epochs))

plt.legend()

plt.grid()

plt.show()