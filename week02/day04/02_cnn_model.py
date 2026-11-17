import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1
        )

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        # 激活函数
        self.relu = nn.ReLU()

        # 池化
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接层
        self.fc = nn.Linear(
            32 * 56 * 56,
            10
        )

    def forward(self, x):

        # 第一层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二层
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # 全连接
        x = self.fc(x)

        return x


# 创建模型
model = CNN()

print(model)


# 模拟输入
x = torch.rand(1, 3, 224, 224)

# 前向传播
output = model(x)

print("输入:", x.shape)
print("输出:", output.shape)