import torch
import torch.nn as nn


# =========================
# 1. 输入图片
# =========================

x = torch.rand(1, 3, 224, 224)

print("输入:", x.shape)


# =========================
# 2. 第一层卷积
# =========================

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)

x = conv1(x)

print("Conv1:", x.shape)


# =========================
# 3. ReLU
# =========================

relu = nn.ReLU()

x = relu(x)

print("ReLU1:", x.shape)


# =========================
# 4. 最大池化
# =========================

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

x = pool(x)

print("Pool1:", x.shape)


# =========================
# 5. 第二层卷积
# =========================

conv2 = nn.Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    padding=1
)

x = conv2(x)

print("Conv2:", x.shape)


# =========================
# 6. ReLU
# =========================

x = relu(x)

print("ReLU2:", x.shape)


# =========================
# 7. 第二次池化
# =========================

x = pool(x)

print("Pool2:", x.shape)


# =========================
# 8. Flatten
# =========================

x = torch.flatten(x, 1)

print("Flatten:", x.shape)


# =========================
# 9. 全连接层
# =========================

fc = nn.Linear(100352, 10)

x = fc(x)

print("Linear:", x.shape)