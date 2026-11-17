import torch
import torch.nn as nn


# =========================
# 1. 创建原始输入
# =========================
x = torch.randn(4, 64, 56, 56)

print("原始输入 x：")
print(x.shape)


# =========================
# 2. 创建 Downsample
# =========================
downsample = nn.Sequential(
    nn.Conv2d(
        64,
        128,
        kernel_size=1,
        stride=2,
        bias=False
    ),
    nn.BatchNorm2d(128)
)


# =========================
# 3. 对 Shortcut 进行变换
# =========================
identity = downsample(x)

print("\n经过 Downsample 后的 Shortcut：")
print(identity.shape)


# =========================
# 4. 模拟主分支 F(x)
# =========================
out = torch.randn(4, 128, 28, 28)

print("\n主分支 F(x)：")
print(out.shape)


# =========================
# 5. 进行残差相加
# =========================
result = out + identity

print("\n相加后的结果：")
print(result.shape)