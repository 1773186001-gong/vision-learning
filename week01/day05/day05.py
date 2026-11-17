import cv2

# 1. 读取图片
image = cv2.imread("test.jpg")

# 2. 转成灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 二值化
ret, binary = cv2.threshold(
    gray,
    50,
    255,
    cv2.THRESH_BINARY
)

# 4. 显示
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()