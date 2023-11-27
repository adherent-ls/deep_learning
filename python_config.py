import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import SequentialSampler

from base.metric.base_post import BaseMetric
from data.dataset.lmdb_dataset_filter import LmdbDatasetFilter
from data.transforms.batch.collate import ImageCollate, LabelCTCCollate
from data.transforms.batch.keep_batch_keys import KeepKeyTensor
from data.transforms.batch.load_batch_keys import LoadBatchKeys
from data.transforms.single.images.image_buffer_decode import ImageBufferDecode
from data.transforms.single.images.image_normal import ZeroMeanNormal
from data.transforms.single.images.image_reshape import ImageReshape
from data.transforms.single.images.image_resize import Resize
from data.transforms.single.load_keys import LoadKeys
from data.transforms.single.texts.text_label_coder import LabelCoder
from data.transforms.single.texts.text_stream_decode import TextStreamDecode
from losses.loss_fn.loss_proxy import CTCLossProxy
from metric.decoder.text_decoder import TextDecoder
from metric.metric.text_metric import TextMetric
from models.modules.self.backbone.res_adapt import ResNetAdapt
from models.modules.self.backbone.text_encoder import Transformer
from models.modules.self.head.fc import FCPrediction
from models.modules.self.neck.reshape import Reshape
from optimizers.lr_scheduler.warm_up import Warmup


class Model(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.backbone = ResNetAdapt(input_channel=3,
                                    output_channel=512)
        self.neck1 = Reshape(transpose_index=[0, 3, 2, 1],
                             reshape_size=[-1, -1, 512])
        self.neck2 = Transformer(width=512,
                                 layers=2,
                                 heads=16)
        self.head = FCPrediction(in_channel=512,
                                 n_class=n_class,
                                 ouk='text_prop')

    def forward(self, data):
        x = self.backbone(data)
        x = self.neck1(x)
        x = self.neck2(x)
        x = self.head(x)
        return x


class OptimizerWithScheduler(object):
    def __init__(self, optimizer, scheduler=None):
        super(OptimizerWithScheduler, self).__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
        else:
            lr = self.base_lr
        return lr

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)


class BuildTransform(object):
    def __init__(self, trans):
        super(BuildTransform, self).__init__()
        self.trans = trans

    def __call__(self, data):
        for item in self.trans:
            data = item(data)
            if data is None:
                return None
        return data


class MutilDataLoader(object):
    def __init__(self, datasets, collate_fn, num_workers, batch_size, shuffle):
        self.shuffle = shuffle
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
        if self.shuffle:
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


class IndexMetric(BaseMetric):
    def __init__(self):
        super(IndexMetric, self).__init__()

        self.acc = 0
        self.normal = 0
        self.t_n = 0
        self.t = 0

    def reset(self):
        self.acc = 0
        self.normal = 0
        self.t_n = 0
        self.t = 0

    def __call__(self, word_predication, word):
        _, index = torch.max(word_predication, dim=-1)
        # pred_texts = self.decoder(index=index)
        # labels = self.decoder(index=word)
        for text, label in zip(index, word):
            label = label[label != 0]
            text = text[text != 0]
            if len(text) != 0:
                text = text[:len(label)]
                if (text == label).all():
                    self.acc += 1
                self.normal += (text == label).sum()
            self.t_n += max(len(label), len(text))
            self.t += 1
        return self.normal / self.t_n

    def once(self, word_predication, word):
        acc = 0
        normal = 0
        t_n = 0
        t = 0
        _, index = torch.max(word_predication, dim=-1)
        for text, label in zip(index, word):
            label = label[label != 0]
            text = text[text != 0]
            if len(text) != 0:
                print(text, label)
                minl = min(len(label), len(text))
#                 if (text[:minl] == label[:minl]).all():
#                     acc += 1
                normal += (text[:minl] == label[:minl]).sum()
            t_n += max(len(label), len(text))
            t += 1
        return normal / t_n

    def value(self):
        return 1 - self.normal / self.t_n


def build_dataloader(root, characters, max_length, batchsize):
    single = BuildTransform([LoadKeys(keys=['images', 'text']),
                             ImageBufferDecode(),
                             Resize(max_size=[-1, 32],
                                    ink=['images', 'text'],
                                    ouk=['images_pad', 'images', 'text']),
                             ZeroMeanNormal(scale=1. / 255.,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                             ImageReshape(permute_indices=[2, 0, 1]),
                             TextStreamDecode.initialization(TextStreamDecode, ink='text'),
                             LabelCoder(characters=characters,
                                        ink='text')])
    batch = BuildTransform([LoadBatchKeys(keys=['images', 'text']),
                            ImageCollate(padding=10,
                                         ink='images'),
                            LabelCTCCollate(max_length=max_length,
                                            ink='text',
                                            ouk=['text', 'lengths']),
                            KeepKeyTensor(keep_data_keys={'images': 'float32'},
                                          keep_label_keys={'text': 'long', 'lengths': 'long'})])

    dataset = LmdbDatasetFilter.initialization(LmdbDatasetFilter, root=root, transforms=single)

    return MutilDataLoader([dataset], batch, 8, batchsize, True)


def train():
    characters = ['blank'] + [item.strip() for item in
                              open('vocab/radical/word_v2.txt', 'r', encoding='utf-8').readlines()] + ['end']
    max_length = 50
    device = 'cuda'
    batchsize = 24

    train_dataset = build_dataloader(
        root='/home/data/data_old/lmdb/train_ship_rec_data/ch_street/train_tight',
        characters=characters,
        max_length=max_length,
        batchsize=batchsize
    )

    val_dataset = build_dataloader(
        root='/home/data/data_old/lmdb/valid_ship_rec_data/ch_street/val_tight',
        characters=characters,
        max_length=max_length,
        batchsize=batchsize
    )

    model = Model(n_class=len(characters)).to(device)
    loss_fn = CTCLossProxy(zero_infinity=True)
    optim_obj = torch.optim.Adam(
        params=model.parameters(),
        lr=0.0001
    )
    sch_obj = Warmup(optim_obj)
    optim = OptimizerWithScheduler(optim_obj, sch_obj)
    # text = TextMetric(decoder=TextDecoder(characters=characters))
    text = IndexMetric()

    step = 0
    for e in range(100):
        bar = tqdm.tqdm(train_dataset)
        for input_data, labels in bar:
            if input_data is None:
                continue
            for k, v in input_data.items():
                v = v.to(device)
                input_data[k] = v
            for k, v in labels.items():
                v = v.to(device)
                labels[k] = v

            preds = model(input_data)

            loss = loss_fn(preds['text_prop'], [labels['text'], labels['lengths']])
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 10 == 0:
                acc = text.once(preds['text_prop'], labels['text'])
                bar.set_postfix_str(f'step:{step}, loss:{float(loss)}, metric:{acc}')
            step += 1


if __name__ == '__main__':
    train()
