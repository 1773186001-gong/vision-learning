import cv2

image = cv2.imread("test.jpg")

print("原来的尺寸：", image.shape)

resize_image = cv2.resize(image, (300, 300))

print("调整后的尺寸：", resize_image.shape)

cv2.imshow("Original", image)
cv2.imshow("Resize", resize_image)

cv2.waitKey(0)
cv2.destroyAllWindows() 