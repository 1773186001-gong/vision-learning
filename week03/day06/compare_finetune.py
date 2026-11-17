import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================
# 1. 基本设置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备：", device)

train_dir = r"D:\python\test\vision-learning\data\train"
test_dir = r"D:\python\test\vision-learning\data\test"


# =========================
# 2. 数据预处理
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# 3. 数据集
# =========================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)


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


print("类别：", train_dataset.classes)
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))


# =========================
# 4. 创建模型
# =========================

def create_model(strategy):

    model = models.resnet18(weights="DEFAULT")

    # 先冻结全部参数
    for param in model.parameters():
        param.requires_grad = False

    # 策略2：解冻 layer4
    if strategy >= 2:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # 策略3：解冻 layer3 + layer4
    if strategy >= 3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # 修改分类层
    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model = model.to(device)

    return model


# =========================
# 5. 训练函数
# =========================

def train_model(strategy):

    model = create_model(strategy)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=0.0001
    )

    print("\n==============================")

    if strategy == 1:
        print("策略1：只训练 FC")

    elif strategy == 2:
        print("策略2：训练 Layer4 + FC")

    else:
        print("策略3：训练 Layer3 + Layer4 + FC")

    print("==============================")


    for epoch in range(5):

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        train_acc = 100 * correct / total


        # =========================
        # 测试
        # =========================

        model.eval()

        test_correct = 0
        test_total = 0


        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                test_total += labels.size(0)

                test_correct += (
                    predicted == labels
                ).sum().item()


        test_acc = (
            100 * test_correct / test_total
        )


        print(
            f"Epoch [{epoch + 1}/5] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Train Accuracy: {train_acc:.2f}% "
            f"Test Accuracy: {test_acc:.2f}%"
        )


# =========================
# 6. 三种策略
# =========================

train_model(1)

train_model(2)

train_model(3)