import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1. 数据预处理
# =========================

transform = transforms.Compose([
    transforms.ToTensor()
])


# =========================
# 2. 加载数据集
# =========================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# =========================
# 3. DataLoader
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================
# 4. CNN
# =========================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.fc = nn.Linear(
            32 * 7 * 7,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# =========================
# 5. 创建模型
# =========================

model = CNN()


# =========================
# 6. Loss
# =========================

loss_fn = nn.CrossEntropyLoss()


# =========================
# 7. Optimizer
# =========================

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


# =========================
# 8. 训练
# =========================

epochs = 3

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # 前向传播
        outputs = model(images)

        # Loss
        loss = loss_fn(outputs, labels)

        # 清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计 Loss
        total_loss += loss.item()

        # 统计准确率
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )


# =========================
# 9. 测试
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


test_accuracy = correct / total

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)