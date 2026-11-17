import cv2

image = cv2.imread("test.jpg")

print("图片shape：", image.shape)
print("图片dtype：", image.dtype)