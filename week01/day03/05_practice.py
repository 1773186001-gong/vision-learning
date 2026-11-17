import numpy as np


# =========================
# 1. 创建一张 10×10 的黑色灰度图
# =========================

image = np.zeros((10, 10), dtype=np.uint8)

print("原始图片：")
print(image)

print("图片 shape：", image.shape)
print("图片 dtype：", image.dtype)


# =========================
# 2. 把中间 4×4 区域设置成 255
# =========================

image[3:7, 3:7] = 255

print("\n修改后的图片：")
print(image)


# =========================
# 3. 查看图片的统计信息
# =========================

print("\n图片信息：")
print("shape：", image.shape)
print("dtype：", image.dtype)
print("最大值：", image.max())
print("最小值：", image.min())
print("平均值：", image.mean())


# =========================
# 4. 创建一张 10×10×3 的 RGB 图片
# =========================

rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)

print("\nRGB图片：")
print("shape：", rgb_image.shape)
print("dtype：", rgb_image.dtype)


# =========================
# 5. 设置三个像素
# =========================

# 中心位置设置为红色
rgb_image[5, 5] = [255, 0, 0]

# 左上角设置为绿色
rgb_image[0, 0] = [0, 255, 0]

# 右下角设置为蓝色
rgb_image[9, 9] = [0, 0, 255]


print("\n三个像素设置完成：")

print("左上角：", rgb_image[0, 0])
print("中心：", rgb_image[5, 5])
print("右下角：", rgb_image[9, 9])