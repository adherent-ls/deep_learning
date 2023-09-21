import numpy as np
from torch.utils.data import SequentialSampler

from utils.register import register


class DataTransform(object):
    def __init__(self, config):
        super(DataTransform, self).__init__()
        self.trans = self.build_transform(config)

    def build_transform(self, config):
        trans = []
        for item_config in config:
            k = item_config['name']
            v = item_config['args']
            trans.append(register.build_from_config(k, v, 'transform'))
        return trans

    def __call__(self, data):
        for item in self.trans:
            data = item(data)
        return data


class MutilDataLoader(object):
    def __init__(self, config):
        self.shuffle = config['shuffle']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.datasets = self.build_dataset(config['Dataset'])
        self.collate_fn = self.build_transform(config['Transforms'])

        sampler = []
        data = []
        for index, dataset in enumerate(self.datasets):
            sampler_item = SequentialSampler(dataset)
            data.append(dataset)
            sampler_item = list(sampler_item)
            sampler.extend([[index, item] for item in sampler_item])
        self.sampler = np.array(sampler)
        if self.shuffle:
            np.random.shuffle(self.sampler)
        self.data = data

    def build_transform(self, config):
        return DataTransform(config)

    def build_filter(self, config):
        filters = []
        for filter_item in config:
            f_k = filter_item['name']
            f_v = filter_item['args']
            filter = register.build_from_config(f_k, f_v, 'filter')
            filters.append(filter)
        return filters

    def build_dataset(self, config):
        datasets = []
        for dataset_item in config:
            k = dataset_item['name']
            v = dataset_item['args']
            if 'filter' in v:
                v['filter'] = self.build_filter(v['filter'])
            datasets.append(register.build_from_config(k, v, 'data'))
        return datasets

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
