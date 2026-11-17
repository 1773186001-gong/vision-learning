import os
import random

import cv2
import torch
import torch.nn as nn

from matplotlib import pyplot as plt


# ============================================================
# 1. CNN
# ============================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
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
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(
            32 * 56 * 56,
            2
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x


# ============================================================
# 2. 创建模型
# ============================================================

model = CNN()


# ============================================================
# 3. 加载模型
# ============================================================

model.load_state_dict(
    torch.load(
        "cat_dog_cnn.pth",
        weights_only=True
    )
)

model.eval()

print("模型加载成功！")


# ============================================================
# 4. 准备测试图片
# ============================================================

classes = [
    "cats",
    "dogs"
]


# 收集所有测试图片
image_list = []

for label, class_name in enumerate(classes):

    class_dir = os.path.join(
        "data/test",
        class_name
    )

    for filename in os.listdir(class_dir):

        image_path = os.path.join(
            class_dir,
            filename
        )

        image_list.append(
            (image_path, label)
        )


print(
    "测试图片数量:",
    len(image_list)
)


# ============================================================
# 5. 随机选择一张图片
# ============================================================

image_path, true_label = random.choice(
    image_list
)

print(
    "图片路径:",
    image_path
)

print(
    "真实类别:",
    classes[true_label]
)


# ============================================================
# 6. 读取图片
# ============================================================

image = cv2.imread(
    image_path
)

if image is None:

    raise FileNotFoundError(
        f"图片读取失败: {image_path}"
    )


# 保存一份用于显示的图片
display_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 7. 图片预处理
# ============================================================

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    image,
    (224, 224)
)

image = torch.from_numpy(
    image
)

# HWC → CHW
image = image.permute(
    2,
    0,
    1
)

# 0~255 → 0~1
image = image.float() / 255.0

# 增加 Batch 维度
image = image.unsqueeze(0)


print(
    "输入 Tensor:",
    image.shape
)


# ============================================================
# 8. CNN预测
# ============================================================

with torch.no_grad():

    output = model(image)

    predicted = output.argmax(
        dim=1
    ).item()


# ============================================================
# 9. 输出结果
# ============================================================

print(
    "模型预测:",
    classes[predicted]
)


if predicted == true_label:

    print("结果: ✅ 预测正确")

else:

    print("结果: ❌ 预测错误")


# ============================================================
# 10. 显示图片
# ============================================================

plt.imshow(
    display_image
)

plt.title(
    f"True: {classes[true_label]} | "
    f"Pred: {classes[predicted]}"
)

plt.axis("off")

plt.show()