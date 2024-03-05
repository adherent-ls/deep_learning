import numpy as np
from torch.utils.data import DataLoader, SequentialSampler


class MutilDataLoader(object):
    def __init__(self, datasets, batch_size, collate_fn, shuffle=True, num_workers=8):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.collate_fn = collate_fn

        sampler = []
        data = []
        for index, dataset in enumerate(self.datasets):
            sampler_item = SequentialSampler(dataset)
            data.append(dataset)
            sampler_item = list(sampler_item)
            sampler.extend([[index, item] for item in sampler_item])
        self.sampler = np.array(sampler)
        if shuffle:
            np.random.shuffle(self.sampler)
        self.data = data

    def __iter__(self):
        for st in range(0, len(self)):
            st = st * self.batch_size
            end = st + self.batch_size
            samples = self.sampler[st:end]
            data = []
            for index, data_index in samples:
                data.append(self.data[index][data_index])
            yield self.collate_fn(data)

    def __len__(self):
        if len(self.sampler) % self.batch_size == 0:
            return len(self.sampler) // self.batch_size
        else:
            return len(self.sampler) // self.batch_size + 1
