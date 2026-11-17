import cv2

image = cv2.imread("test.jpg")

print("原始图片：")
print("shape =", image.shape)
print("dtype =", image.dtype)


# 1. 裁剪
crop = image[100:300, 100:300]


# 2. 缩放
resize = cv2.resize(image, (300, 300))


# 3. 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 4. 保存
cv2.imwrite("crop.jpg", crop)
cv2.imwrite("resize.jpg", resize)
cv2.imwrite("gray.jpg", gray)


# 5. 显示
cv2.imshow("Original", image)
cv2.imshow("Crop", crop)
cv2.imshow("Resize", resize)
cv2.imshow("Gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()