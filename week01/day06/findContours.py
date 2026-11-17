import cv2

# 读取图片
image = cv2.imread("test.jpg")

# 灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# 查找轮廓
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("找到的轮廓数量：", len(contours))

# 在原图上画轮廓
result = image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imshow("Original", image)
cv2.imshow("Binary", binary)
cv2.imshow("Contours", result)

cv2.waitKey(0)
cv2.destroyAllWindows()