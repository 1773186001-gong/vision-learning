import cv2
import torch

from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):

    def __init__(self):
        self.images = [
            "test01.png",
            "test02.png"
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # 1. 获取图片路径
        img_path = self.images[index]

        # 2. 读取图片
        img = cv2.imread(img_path)

        # 3. 检查图片是否读取成功
        if img is None:
            raise FileNotFoundError(
                f"图片读取失败: {img_path}"
            )

        # 4. 打印原始图片尺寸
        print(
            f"{img_path} 原始尺寸: {img.shape}"
        )

        # 5. 统一图片尺寸
        img = cv2.resize(
            img,
            (224, 224)
        )

        # 6. numpy -> Tensor
        img = torch.from_numpy(img)

        # 7. HWC -> CHW
        img = img.permute(
            2,
            0,
            1
        )

        return img


# 创建 Dataset
dataset = ImageDataset()

print("数据集数量:", len(dataset))


# 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# 读取一个 batch
for imgs in loader:

    print("Batch shape:")
    print(imgs.shape)