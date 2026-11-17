from PIL import Image
from torchvision import transforms

img = Image.open("test01.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

tensor_img = transform(img)

print("shape:", tensor_img.shape)
print("min:", tensor_img.min())
print("max:", tensor_img.max())