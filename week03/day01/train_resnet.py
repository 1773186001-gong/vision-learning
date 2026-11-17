import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1. Residual Block
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 核心：残差连接
        out = out + identity

        out = self.relu(out)

        return out


# =========================
# 2. ResNet
# =========================
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# =========================
# 3. 数据
# =========================
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64
)


# =========================
# 4. 模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# 5. 简短训练
# =========================
for epoch in range(3):

    model.train()

    total = 0
    correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100

    print(
        f"Epoch [{epoch + 1}/3] "
        f"Loss: {loss.item():.4f} "
        f"Train Accuracy: {accuracy:.2f}%"
    )


# =========================
# 6. 测试
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

print(f"Test Accuracy: {correct / total * 100:.2f}%")