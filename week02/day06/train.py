import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Dataset
# ============================================================

class MyDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        # 0 = cats
        # 1 = dogs
        self.classes = ["cats", "dogs"]

        self.images = []

        # 遍历 cats 和 dogs 文件夹
        for label, class_name in enumerate(self.classes):

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            # 遍历文件夹中的所有图片
            for filename in os.listdir(class_dir):

                image_path = os.path.join(
                    class_dir,
                    filename
                )

                # 保存：图片路径 + 标签
                self.images.append(
                    (image_path, label)
                )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        # 获取图片路径和标签
        image_path, label = self.images[index]

        # 读取图片
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"图片读取失败: {image_path}"
            )

        # OpenCV:
        # BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # 统一图片大小
        image = cv2.resize(
            image,
            (224, 224)
        )

        # NumPy → Tensor
        image = torch.from_numpy(image)

        # HWC → CHW
        image = image.permute(2, 0, 1)

        # uint8 0~255 → float 0~1
        image = image.float() / 255.0

        # 标签转换成 Tensor
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# 2. CNN
# ============================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # 激活函数
        self.relu = nn.ReLU()

        # 最大池化
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 全连接层
        #
        # 224 × 224
        # ↓ 第一次池化
        # 112 × 112
        # ↓ 第二次池化
        # 56 × 56
        #
        # 通道数 = 32
        #
        # 所以：
        # 32 × 56 × 56 = 100352
        #
        # 最后输出 2 个类别：
        # 0 = cats
        # 1 = dogs
        self.fc = nn.Linear(
            32 * 56 * 56,
            2
        )

    def forward(self, x):

        # 第一层卷积
        x = self.conv1(x)

        # ReLU
        x = self.relu(x)

        # 池化
        x = self.pool(x)

        # 第二层卷积
        x = self.conv2(x)

        # ReLU
        x = self.relu(x)

        # 池化
        x = self.pool(x)

        # 展平
        x = torch.flatten(
            x,
            1
        )

        # 全连接层
        x = self.fc(x)

        return x


# ============================================================
# 3. 创建训练集
# ============================================================

train_dataset = MyDataset(
    "data/train"
)

print("训练集数量:", len(train_dataset))


# ============================================================
# 4. 创建测试集
# ============================================================

test_dataset = MyDataset(
    "data/test"
)

print("测试集数量:", len(test_dataset))


# ============================================================
# 5. 创建 DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# ============================================================
# 6. 测试 Dataset
# ============================================================

image, label = train_dataset[0]

print(
    "单张图片 shape:",
    image.shape
)

print(
    "单张图片标签:",
    label
)


# ============================================================
# 7. 创建 CNN
# ============================================================

model = CNN()

print("\n模型创建完成")


# ============================================================
# 8. Loss
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 9. Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. 开始训练
# ============================================================

epochs = 10

print("\n========== 开始训练 ==========")

for epoch in range(epochs):

    # 设置为训练模式
    model.train()

    total_loss = 0

    correct = 0
    total = 0

    # 遍历 Batch
    for images, labels in train_loader:

        # ----------------------------------------------------
        # ① CNN预测
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # ② 计算 Loss
        # ----------------------------------------------------

        loss = loss_fn(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # ③ 清空旧梯度
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # ④ 反向传播
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # ⑤ 更新模型参数
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # 统计 Loss
        # ----------------------------------------------------

        total_loss += loss.item()

        # ----------------------------------------------------
        # 统计准确率
        # ----------------------------------------------------

        predicted = outputs.argmax(
            dim=1
        )

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    # 当前 Epoch 的准确率
    train_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2%}"
    )


# ============================================================
# 11. 测试模型
# ============================================================

print("\n========== 开始测试 ==========")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        # 模型预测
        outputs = model(images)

        # 获取预测类别
        predicted = outputs.argmax(
            dim=1
        )

        # 统计正确数量
        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)


test_accuracy = correct / total

print(
    f"Test Accuracy: {test_accuracy:.2%}"
)


# ============================================================
# 12. 查看一批测试结果
# ============================================================

print("\n========== 查看预测结果 ==========")

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(
            dim=1
        )

        print("真实标签:", labels)
        print("预测标签:", predicted)

        break


# ============================================================
# 13. 保存模型
# ============================================================

torch.save(
    model.state_dict(),
    "cat_dog_cnn.pth"
)

print("\n模型已经保存:")
print("cat_dog_cnn.pth")


# ============================================================
# 14. 单张图片预测
# ============================================================

print("\n========== 单张图片预测 ==========")

# 获取测试集第一张图片
image, label = test_dataset[0]

# 增加 Batch 维度
image = image.unsqueeze(0)

model.eval()

with torch.no_grad():

    output = model(image)

    predicted = output.argmax(
        dim=1
    ).item()


# 标签对应的类别
class_names = [
    "cats",
    "dogs"
]

print(
    "真实类别:",
    class_names[label.item()]
)

print(
    "预测类别:",
    class_names[predicted]
)