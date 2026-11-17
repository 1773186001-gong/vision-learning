from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self):
        self.data = ["A", "B", "C", "D", "E"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


dataset = MyDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for batch in loader:
    print(batch)