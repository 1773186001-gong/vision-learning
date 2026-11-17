import cv2

# 读取
image = cv2.imread("test.jpg")

# 灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny边缘检测
edges = cv2.Canny(gray, 100, 200)

# 显示
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()