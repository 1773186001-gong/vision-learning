from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


# =========================
# 1. 读取图片
# =========================
import os

cats_dir = r"D:\python\test\vision-learning\data\train\cats"

image_files = os.listdir(cats_dir)

image_path = os.path.join(cats_dir, image_files[0])

print("使用的图片：", image_files[0])

image = Image.open(image_path)

print("原始图片尺寸：", image.size)


# =========================
# 2. 定义数据增强
# =========================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    )
])


# =========================
# 3. 生成 6 张增强图片
# =========================
augmented_images = []

for i in range(6):
    augmented_image = transform(image)
    augmented_images.append(augmented_image)


# =========================
# 4. 显示结果
# =========================
plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    plt.imshow(augmented_images[i])

    plt.title(f"Augmented {i + 1}")

    plt.axis("off")


plt.tight_layout()
plt.show()