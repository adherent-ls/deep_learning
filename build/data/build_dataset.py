from torch.utils.data import Dataset


class BuildDataset(Dataset):
    def __init__(self, dataset, filters, transforms=None):
        super().__init__()
        if filters is not None:
            indices = filters(dataset)
        else:
            nSamples = int(len(dataset))
            indices = [i for i in range(nSamples)]
        self.indices = indices

        self.dataset = dataset
        self.transforms = transforms
        print(len(self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        data = self.dataset[int(self.indices[item])]
        if self.transforms is not None:
            data = self.transforms(data)
        return data
