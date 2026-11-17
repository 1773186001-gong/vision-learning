import torch
from torchvision import models


# ========================================
# 1. 随机初始化的 ResNet18
# ========================================

model_random = models.resnet18(weights=None)


# ========================================
# 2. 使用预训练权重的 ResNet18
# ========================================

model_pretrained = models.resnet18(weights="DEFAULT")


# ========================================
# 3. 查看第一层卷积的参数
# ========================================

random_weight = model_random.conv1.weight

pretrained_weight = model_pretrained.conv1.weight


print("随机初始化模型：")
print("第一层卷积参数前5个：")
print(random_weight.view(-1)[:5])


print("\n预训练模型：")
print("第一层卷积参数前5个：")
print(pretrained_weight.view(-1)[:5])


# ========================================
# 4. 比较参数
# ========================================

difference = torch.mean(
    torch.abs(random_weight - pretrained_weight)
)

print("\n两个模型第一层参数平均差异：")
print(difference.item())