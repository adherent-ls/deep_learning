import os

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import DataParallel
from tqdm import tqdm

from build.criterion import criterion_instance
from build.data import dataloader_instance
from build.data.build_dataset import BuildDataset
from build.data.build_filter import BuildFilter
from build.data.build_transform import BuildTransform
from build.metric import metric_instance, decoder_instance
from build.model import module_instance
from build.model.build_data_parallel import BuildDataParallel
from build.model.build_module import BuildModule
from build.optimizer import optim_instance
from data.collate.image.image_padding import ImageCollate
from data.collate.label.label_padding import LabelCollate
from data.collate.split.batch_split import BatchSplit
from data.dataloader.multi_dataloader import MutilDataLoader
from data.dataset.lmdb_dataset import LmdbDataset
from data.filter.image_filter import LmdbImageFilter
from data.filter.label_filter import LmdbOutVocabFilter
from data.collate.convert.numpy_to_tensor import NPToTensor
from data.transforms.image.image_normal import ImageNormal
from data.transforms.image.image_pil_to_np import ImagePilToNP
from data.transforms.image.image_reshape import ImageReshape
from data.transforms.image.image_resize import ImageResizeNormal
from data.transforms.image.lmdb_image_decode import LmdbImageDecode
from data.transforms.label.lmdb_stream_decode import LmdbStreamDecode
from data.transforms.label.text_encode import TextEncode
from models.losses.loss_fn.loss_proxy import CTCLossProxy
from models.metrics.decoder.text_decoder import TextDecoder
from models.metrics.metric.rec_metric_paddle import RecMetric
from models.modules.backbone.cnn.res_adapt import ResNet
from models.modules.head.text_recognize.ctc import CTC
from models.modules.neck.rnn.bilstm_paddle import EncoderWithRNN
from models.networks.CRNN import CRNN
from models.optimizers.lr_scheduler.warm_up import Warmup
from models.optimizers.optimizer_with_scheduler import OptimizerWithScheduler


def main():
    if torch.cuda.is_available():
        # 设置默认的 GPU 设备为第一个 GPU（设备索引为0）
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        # 如果没有可用的 GPU 设备，使用 CPU
        device = torch.device("cpu")
        print("No GPU available, using CPU.")

    data_root = r'/home/data/data_old/MJSynth'

    charactor = [x.strip('\n') for x in open('vocab/word/synth_text_vocab', 'r', encoding='utf-8').readlines()]
    charactor = ['blank'] + charactor + ['end']
    max_length = 25
    save_path = '/home/data/workspace/training_models/deep_learning/deep_learning_check_v2'
    yaml_path = r'config/text_recognizer/CRNN-BiLSTM-CTC-V1.yml'
    lr = 0.0005
    batch_size = 8
    split_step = 100
    print_step = 100
    eval_step = 10000
    epoch = 100

    os.makedirs(save_path, exist_ok=True)

    config = yaml.load(open(yaml_path, 'r', encoding='utf-8'), Loader=yaml.Loader)

    train_dataloader = dataloader_instance(config['data']['Train'])
    # valid_dataloader = dataloader_instance(config['data']['Eval'])

    model = module_instance(config['model'])
    model.to(device)
    # model.resume_model(os.path.join(save_path, 'latest.pth'))

    # 如果有多个 GPU，使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = BuildDataParallel(model, device_ids=[0, 1])
        print(f"Using {torch.cuda.device_count()} GPUs.")

    optimizer_with_scheduler = optim_instance(model, config['optimizer'])

    criterion = criterion_instance(config['criterion'])
    metric = metric_instance(config['metric'])

    decoder = decoder_instance(config['decoder'])

    index = 0
    loss_v = 0
    epoch = config['epoch']
    for i in range(epoch):
        bar = tqdm(train_dataloader)
        for item in bar:
            images, labels = item
            texts, lengths = labels
            images = images.to(device)
            texts = texts.to(device)
            lengths = lengths.to(device)

            preds = model(images)

            loss = criterion(preds, [texts, lengths])
            optimizer_with_scheduler.zero_grad()
            loss.backward()
            curr_lr = optimizer_with_scheduler.step()

            _, pred_indices = preds.max(dim=2)
            preds, labels = decoder(pred_indices), decoder(texts)
            metric(preds, labels)

            loss_v += loss
            index += 1
            if index % print_step == 0:
                curr_lr = np.round(curr_lr, 7)
                loss_v = np.round(float(loss_v) / print_step, 5)
                acc = metric.get_metric()
                bar.set_postfix_str(f'{index},{curr_lr},{loss_v},{acc}')
                metric.reset()
                loss_v = 0
            if index % eval_step == 0:
                model.save_model('latest', metric.get_metric())


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
