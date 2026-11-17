import cv2
import torch


img = cv2.imread("test.png")


tensor_img=torch.from_numpy(img)


print("转换前:")
print(tensor_img.shape)



tensor_img=tensor_img.permute(
    2,
    0,
    1
)


print("转换后:")
print(tensor_img.shape)