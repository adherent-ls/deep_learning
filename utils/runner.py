import time

import Levenshtein
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.collate.align import AlignCollate
from data.collate.ctc import CTCLabelConverter
from data.dataset.lmdb_dataset import LmdbDataset
from data.filter.filter import LabelVocabCheckFilter, ImageCheckFilter
from data.filter.text_data_filter import TextDataFilter
from data.transforms.image_resize import ImageResize
from models.losses.loss_fn.loss_proxy import CTCLossProxy
from models.metrics.decoder.text_decoder import TextDecoder
from models.metrics.metric.rec_metric_paddle import RecMetric
from models.metrics.metric.text_metric import TextMetric
from models.networks.CRNN import CRNN
from models.optimizers.lr_scheduler.warm_up import Warmup
from models.optimizers.optimizer_with_scheduler import OptimizerWithScheduler

save_path = r'/home/data/workspace/training_models/deep_learning/restruct'
chars = [item.strip('\n') for item in open('/home/sy_lw/Code/self/deep_learning/vocab/radical/word_v2.txt').readlines()]
config = {
    'model': {
        'input_channel': 3,
        'mid_channel': 480,
        'output_channel': 512,
        'num_class': len(chars),
        'save_path': save_path
    },
    'data': {
        'train_path': r'/home/data/data_old/lmdb/train_ship_rec_data/ch_street/train_tight',
        'valid_path': r'/home/data/data_old/lmdb/valid_ship_rec_data/ch_street/val_tight',
        'batch_size': 128,
        'workers': 8,
        'character': chars,
        'batch_max_length': 25,
        'imgH': 32,
        'imgW': 256,
        'PAD': True,
    },
    'optim': {
        'lr': 0.001,
        'beta': 0.9
    }
}


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Runner(object):
    def __init__(self):
        super().__init__()
        mcg = config['model']

        self.model = CRNN(mcg['input_channel'],
                          mcg['mid_channel'],
                          mcg['output_channel'],
                          mcg['num_class'],
                          mcg['save_path'])

        dcg = config['data']
        self.batch_max_length = dcg['batch_max_length']

        transforms = ImageResize(imgH=dcg['imgH'], imgW=dcg['imgW'], keep_ratio_with_pad=dcg['PAD'])
        self.train_dataloader = self.build_dataset(dcg['train_path'], dcg['batch_size'], dcg['workers'], transforms)
        self.valid_dataloader = self.build_dataset(dcg['valid_path'], dcg['batch_size'], dcg['workers'], transforms)
        print(len(self.train_dataloader), len(self.valid_dataloader))

        self.converter = CTCLabelConverter(character=dcg['character'], batch_max_length=dcg['batch_max_length'])

        self.criterion = CTCLossProxy(zero_infinity=True)

        ocg = config['optim']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=ocg['lr'], betas=(ocg['beta'], 0.999),
                                     weight_decay=0.00004)
        scheduler = Warmup(optimizer=optimizer, warm=5 * len(self.train_dataloader))
        self.optimizer_with_scheduler = OptimizerWithScheduler(optimizer=optimizer, scheduler=scheduler)
        self.device = 'cuda:0'

        self.metric = RecMetric()

    def build_dataset(self, path, batch_size, workers, transforms):
        filters = TextDataFilter(
            filters=[
                LabelVocabCheckFilter(label_length_limit=self.batch_max_length, characters=chars),
                ImageCheckFilter()
            ],
            recache=False
        )
        dataset = LmdbDataset(path, filters, transforms)
        valid_loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=workers,
            collate_fn=None, pin_memory=True)
        return valid_loader

    def train(self):
        iteration = 0
        self.model.to(self.device)
        st = time.time()
        loss_avg = Averager()
        for i in range(100):
            for images, labels in self.train_dataloader:
                text, length = self.converter.encode(labels)

                images = images.to(self.device)
                text = text.to(self.device)
                length = length.to(self.device)

                batch_size = images.size(0)

                preds = self.model(images)
                cost = self.criterion(preds, (text, length))

                self.model.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping with 5 (Default)
                lr = self.optimizer_with_scheduler.step()

                _, preds_index = preds.max(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)
                text_label = self.converter.decode(text, length)
                self.metric((preds_str, text_label))
                loss_avg.add(cost)
                if (iteration + 1) % 100 == 0:
                    print(iteration + 1, lr, loss_avg.val(), time.time() - st, self.metric.get_metric())
                    self.metric.reset()
                    st = time.time()
                if (iteration + 1) % 2000 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        valid_loss, norm_ED, infer_time, length_of_data = self.validation()
                        print(valid_loss, norm_ED, infer_time, length_of_data)
                    self.model.save_model('last', norm_ED)
                    self.model.train()
                    loss_avg.reset()
                iteration += 1

    def validation(self):
        """ validation or evaluation """
        self.metric.reset()

        length_of_data = 0
        infer_time = 0
        valid_loss_avg = Averager()

        for i, (image_tensors, labels) in enumerate(self.valid_dataloader):
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(self.device)

            text_for_loss, length_for_loss = self.converter.encode(labels)

            start_time = time.time()
            preds = self.model(image)
            forward_time = time.time() - start_time

            cost = self.criterion(preds, (text_for_loss, length_for_loss))

            _, preds_index = preds.max(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)
            text_label = self.converter.decode(text_for_loss, length_for_loss)
            infer_time += forward_time
            valid_loss_avg.add(cost)
            self.metric((preds_str, text_label))

        return valid_loss_avg.val(), self.metric.get_metric(), infer_time, length_of_data
