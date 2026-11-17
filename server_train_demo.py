
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# 服务器上的固定路径
PROJECT_DIR = Path(r"D:\vision-learning-server")
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 每次运行使用独立日志文件名
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"train_demo_{RUN_ID}.log"
MODEL_FILE = CHECKPOINT_DIR / f"resnet18_demo_{RUN_ID}.pth"


def setup_logger():
    logger = logging.getLogger("server_train_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    logger = setup_logger()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，停止训练。")

        device = torch.device("cuda:0")
        logger.info("PyTorch: %s", torch.__version__)
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("数据目录: %s", DATA_DIR)
        logger.info("日志文件: %s", LOG_FILE)
        logger.info("模型文件: %s", MODEL_FILE)

        if not (DATA_DIR / "train").is_dir():
            raise FileNotFoundError(f"训练目录不存在: {DATA_DIR / 'train'}")
        if not (DATA_DIR / "test").is_dir():
            raise FileNotFoundError(f"测试目录不存在: {DATA_DIR / 'test'}")

        # 预处理：统一尺寸、转换张量并按 ImageNet 参数归一化
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dataset = datasets.ImageFolder(
            DATA_DIR / "train", transform=transform
        )
        test_dataset = datasets.ImageFolder(
            DATA_DIR / "test", transform=transform
        )

        if train_dataset.classes != test_dataset.classes:
            raise ValueError("训练集与测试集类别不一致。")

        logger.info("类别: %s", train_dataset.classes)
        logger.info("训练图片数: %d", len(train_dataset))
        logger.info("测试图片数: %d", len(test_dataset))

        # Windows 上先使用单进程加载数据，降低后台运行复杂度
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=0
        )

        # 优先使用本地缓存的预训练权重；没有缓存时使用随机初始化，
        # 避免后台任务因下载权重而意外等待。
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            logger.info("已加载本地可用的 ResNet18 预训练权重。")
        except Exception as exc:
            logger.warning(
                "无法获取预训练权重，将使用随机初始化：%s",
                exc,
            )
            model = models.resnet18(weights=None)

        for parameter in model.parameters():
            parameter.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(train_dataset.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

        # 训练 3 轮，先验证整条流程
        epochs = 3

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += batch_size

            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # 测试集评估
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    test_correct += (
                        outputs.argmax(dim=1) == labels
                    ).sum().item()
                    test_total += labels.size(0)

            test_acc = test_correct / max(test_total, 1)

            logger.info(
                "Epoch [%d/%d] | Loss: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%%",
                epoch + 1,
                epochs,
                train_loss,
                train_acc * 100,
                test_acc * 100,
            )

        # 保存模型参数、类别映射和训练配置
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "epoch": epochs,
                "model_name": "resnet18",
                "device": str(device),
            },
            MODEL_FILE,
        )

        logger.info("训练完成，模型已保存。")
        logger.info("模型路径: %s", MODEL_FILE)
        logger.info(
            "GPU 峰值显存: %.1f MB",
            torch.cuda.max_memory_allocated() / 1024**2,
        )

    except Exception:
        logger.exception("训练失败。")
        raise

    finally:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
