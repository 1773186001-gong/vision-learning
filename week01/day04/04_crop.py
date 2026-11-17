import cv2

image = cv2.imread("test.jpg")

crop = image[100:300, 100:300]

cv2.imshow("Original", image)
cv2.imshow("Crop", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()