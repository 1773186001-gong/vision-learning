import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# =========================
# 1. 配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备：", device)

model_path = r"D:\python\test\vision-learning\week03\day07\resnet18_day07.pth"

# 这里先填写一张你想测试的图片
image_path = r"D:\python\test\vision-learning\data\test\cats\cat_1.jpg"

classes = ["cats", "dogs"]


# =========================
# 2. 创建模型
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# 加载训练好的参数
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("模型加载成功")


# =========================
# 3. 图片预处理
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# 4. 读取图片
# =========================

image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)

# 增加 batch 维度
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# =========================
# 5. 模型预测
# =========================

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


# =========================
# 6. 输出结果
# =========================

print("预测类别：", classes[predicted_class])

print(
    "预测概率：",
    probabilities[0].tolist()
)