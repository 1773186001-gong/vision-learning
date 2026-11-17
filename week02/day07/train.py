import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# 1. Dataset
# ============================================================

class MyDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        # 0 = cats
        # 1 = dogs
        self.classes = ["cats", "dogs"]

        self.transform = transform

        self.images = []

        # 遍历 cats 和 dogs
        for label, class_name in enumerate(self.classes):

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            # 遍历图片
            for filename in os.listdir(class_dir):

                image_path = os.path.join(
                    class_dir,
                    filename
                )

                self.images.append(
                    (image_path, label)
                )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_path, label = self.images[index]

        # ====================================================
        # 读取图片
        # ====================================================

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise FileNotFoundError(
                f"图片读取失败: {image_path}"
            )

        # ====================================================
        # BGR → RGB
        # ====================================================

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # NumPy → PIL
        # ====================================================

        image = Image.fromarray(
            image
        )

        # ====================================================
        # 数据增强 / 预处理
        # ====================================================

        if self.transform is not None:

            image = self.transform(
                image
            )

        # ====================================================
        # 标签
        # ====================================================

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# 2. 训练集数据增强
# ============================================================

train_transform = transforms.Compose([

    # 随机水平翻转
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # 随机旋转
    transforms.RandomRotation(
        degrees=15
    ),

    # 随机裁剪，然后调整到 224×224
    transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0)
    ),

    # 轻微改变颜色
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    # PIL → Tensor
    # 同时：
    # HWC → CHW
    # 0~255 → 0~1
    transforms.ToTensor()

])


# ============================================================
# 3. 测试集只做普通预处理
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor()

])


# ============================================================
# 4. 创建训练集
# ============================================================

train_dataset = MyDataset(
    "data/train",
    transform=train_transform
)

print(
    "训练集数量:",
    len(train_dataset)
)


# ============================================================
# 5. 创建测试集
# ============================================================

test_dataset = MyDataset(
    "data/test",
    transform=test_transform
)

print(
    "测试集数量:",
    len(test_dataset)
)


# ============================================================
# 6. 创建 DataLoader
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
# 7. CNN + Dropout
# ============================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # 第一层卷积
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # 第二层卷积
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # 最大池化
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=0.5
        )

        # ----------------------------------------------------
        # 全连接层
        # ----------------------------------------------------

        self.fc = nn.Linear(
            32 * 56 * 56,
            2
        )

    def forward(self, x):

        # ====================================================
        # 第一层
        # ====================================================

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        # ====================================================
        # 第二层
        # ====================================================

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        # ====================================================
        # 展平
        # ====================================================

        x = torch.flatten(
            x,
            1
        )

        # ====================================================
        # Dropout
        # ====================================================

        x = self.dropout(x)

        # ====================================================
        # 分类
        # ====================================================

        x = self.fc(x)

        return x


# ============================================================
# 8. 创建模型
# ============================================================

model = CNN()

print(
    "\n模型创建完成"
)


# ============================================================
# 9. Loss
# ============================================================

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# 10. Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 11. 开始训练
# ============================================================

epochs = 10

print(
    "\n========== 开始训练 =========="
)

for epoch in range(epochs):

    # --------------------------------------------------------
    # 训练模式
    # --------------------------------------------------------

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    # --------------------------------------------------------
    # 遍历 Batch
    # --------------------------------------------------------

    for images, labels in train_loader:

        # ====================================================
        # ① 模型预测
        # ====================================================

        outputs = model(images)

        # ====================================================
        # ② 计算 Loss
        # ====================================================

        loss = loss_fn(
            outputs,
            labels
        )

        # ====================================================
        # ③ 清空梯度
        # ====================================================

        optimizer.zero_grad()

        # ====================================================
        # ④ 反向传播
        # ====================================================

        loss.backward()

        # ====================================================
        # ⑤ 更新参数
        # ====================================================

        optimizer.step()

        # ====================================================
        # Loss
        # ====================================================

        total_loss += loss.item()

        # ====================================================
        # 预测类别
        # ====================================================

        predicted = outputs.argmax(
            dim=1
        )

        # ====================================================
        # 统计准确率
        # ====================================================

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    # --------------------------------------------------------
    # 当前 Epoch 训练准确率
    # --------------------------------------------------------

    train_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2%}"
    )


# ============================================================
# 12. 测试模型
# ============================================================

print(
    "\n========== 开始测试 =========="
)

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


# ============================================================
# 13. 测试准确率
# ============================================================

test_accuracy = correct / total

print(
    f"Test Accuracy: {test_accuracy:.2%}"
)


# ============================================================
# 14. 查看一批预测
# ============================================================

print(
    "\n========== 查看预测结果 =========="
)

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(
            dim=1
        )

        print(
            "真实标签:",
            labels
        )

        print(
            "预测标签:",
            predicted
        )

        break


# ============================================================
# 15. 保存模型
# ============================================================

torch.save(
    model.state_dict(),
    "cat_dog_cnn_augmentation.pth"
)

print(
    "\n模型已经保存:"
)

print(
    "cat_dog_cnn_augmentation.pth"
)