import cv2

image = cv2.imread("test.jpg")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# 创建 kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3,3)
)

# 腐蚀
eroded = cv2.erode(
    binary,
    kernel,
    iterations=1
)

# 膨胀
dilated = cv2.dilate(
    binary,
    kernel,
    iterations=1
)

cv2.imshow("Binary", binary)
cv2.imshow("Eroded", eroded)
cv2.imshow("Dilated", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()