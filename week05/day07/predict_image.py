
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


# =========================
# 1. 路径设置
# =========================
BASE_DIR = r"D:\python\test\vision-learning"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "week05",
    "day05",
    "resnet18_finetune.pth"
)

# 默认使用测试集中的一张图片
# 你可以修改为任意猫狗图片的完整路径
IMAGE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test",
    "cats",
    "cat_118.jpg"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# =========================
# 2. 检查文件
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"模型文件不存在：{MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"图片文件不存在：{IMAGE_PATH}"
    )


# =========================
# 3. 加载模型
# =========================
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("类别顺序：", class_names)
print("模型加载成功！")


# =========================
# 4. 图片预处理
# 必须与训练时一致
# =========================
weights = ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# =========================
# 5. 读取并预测图片
# =========================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)

predicted_index = probabilities.argmax(dim=1).item()
predicted_class = class_names[predicted_index]

print("\n========== 预测结果 ==========")
print("图片路径：", IMAGE_PATH)
print("预测类别：", predicted_class)

for i, name in enumerate(class_names):
    probability = probabilities[0, i].item() * 100
    print(f"{name} 概率：{probability:.2f}%")

print("\n提示：预测概率不等于真实正确率。")


# =========================
# 6. 可视化预测结果
# =========================
plt.figure(figsize=(6, 5))
plt.imshow(image)
plt.title(
    f"Predicted: {predicted_class}"
)
plt.axis("off")
plt.tight_layout()
plt.show()
