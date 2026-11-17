from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms


# =========================
# 1. 读取图片
# =========================
image_path = r"D:\python\test\vision-learning\data\train\cats\001.jpg"

image = Image.open(image_path)

print("原始图片尺寸：", image.size)


# =========================
# 2. 创建数据增强
# =========================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(20)
])


# =========================
# 3. 对图片进行数据增强
# =========================
augmented_image = transform(image)


print("增强后的图片尺寸：", augmented_image.size)


# =========================
# 4. 显示图片
# =========================
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")


plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Augmented")
plt.axis("off")


plt.show()