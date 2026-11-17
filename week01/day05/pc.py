import cv2

image = cv2.imread("test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# 边缘检测
edges = cv2.Canny(
    gray,
    100,
    200
)

cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()