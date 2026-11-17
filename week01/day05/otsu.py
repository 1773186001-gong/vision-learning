import cv2

image = cv2.imread("test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("自动计算的阈值：", ret)

cv2.imshow("Gray", gray)
cv2.imshow("Binary", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()