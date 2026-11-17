
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader


# =========================
# 1. 路径与设备
# =========================
BASE_DIR = r"D:\python\test\vision-learning"
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(
    BASE_DIR, "week05", "day05", "resnet18_finetune.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "week05", "day06"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)
print("模型路径：", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"没有找到模型文件：{MODEL_PATH}"
    )


# =========================
# 2. 测试集预处理
# 必须与训练时保持一致
# =========================
weights = ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("类别：", class_names)
print("测试集数量：", len(test_dataset))


# =========================
# 3. 加载训练好的模型
# =========================
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("模型加载成功！")


# =========================
# 4. 预测并记录错误图片
# =========================
all_labels = []
all_predictions = []
all_confidences = []
wrong_samples = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        predictions = predictions.cpu().tolist()
        confidences = confidences.cpu().tolist()
        labels_list = labels.tolist()

        all_labels.extend(labels_list)
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)

        # 通过 ImageFolder 的样本列表取得图片路径
        start_index = len(all_labels) - len(labels_list)

        for i, (true_label, pred_label, confidence) in enumerate(
            zip(labels_list, predictions, confidences)
        ):
            if true_label != pred_label:
                sample_index = start_index + i
                image_path, _ = test_dataset.samples[sample_index]

                wrong_samples.append({
                    "path": image_path,
                    "true": true_label,
                    "pred": pred_label,
                    "confidence": confidence
                })


# =========================
# 5. 输出整体结果
# =========================
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

correct_count = int(
    (all_labels == all_predictions).sum()
)
accuracy = correct_count / len(all_labels) * 100

print("\n========== 测试结果 ==========")
print(f"正确数量：{correct_count}/{len(all_labels)}")
print(f"测试准确率：{accuracy:.2f}%")
print(f"错误图片数量：{len(wrong_samples)}")


# =========================
# 6. 逐张输出错误图片信息
# =========================
print("\n========== 错误图片详情 ==========")

if not wrong_samples:
    print("本次测试没有发现误分类图片。")

for index, item in enumerate(wrong_samples, start=1):
    print(f"\n错误图片 {index}")
    print("图片路径：", item["path"])
    print("真实类别：", class_names[item["true"]])
    print("预测类别：", class_names[item["pred"]])
    print(f"预测类别概率：{item['confidence']:.2%}")


# =========================
# 7. 生成误分类图片汇总图
# =========================
if wrong_samples:
    columns = 2
    rows = math.ceil(len(wrong_samples) / columns)

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(10, 4 * rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, wrong_samples):
        image = Image.open(item["path"]).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            f"True: {class_names[item['true']]}\n"
            f"Pred: {class_names[item['pred']]}, "
            f"Prob: {item['confidence']:.1%}"
        )
        ax.axis("off")

    for ax in axes[len(wrong_samples):]:
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, "misclassified_images.png"
    )
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\n错误图片汇总图已保存：", output_path)

print("\n误分类分析完成！")
