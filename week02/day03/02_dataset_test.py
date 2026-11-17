import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self):

        self.images = [
            "test01.png",
            "test02.png"
        ]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = self.images[index]

        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(img_path)

        img = self.transform(img)

        return img


dataset = ImageDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for imgs in loader:
    print("Batch:", imgs.shape)