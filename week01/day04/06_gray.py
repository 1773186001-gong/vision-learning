import cv2

image = cv2.imread("test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("原图shape：", image.shape)
print("灰度图shape：", gray.shape)

cv2.imshow("Original", image)
cv2.imshow("Gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()