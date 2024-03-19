import os
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
from tqdm import tqdm

from build.data.build_dataset import BuildDataset
from build.data.build_filter import BuildFilter
from build.data.build_transform import BuildTransform
from build.model.build_data_parallel import BuildDataParallel
from build.model.build_module import BuildModule
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
from models.losses.loss_fn.rlhf_loss import RLHFLossProxy
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
        device1 = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        # 如果没有可用的 GPU 设备，使用 CPU
        device = torch.device("cpu")
        device1 = torch.device("cpu")
        print("No GPU available, using CPU.")

    data_root = r'/home/liuwei/workspace/data/MJSynth'
    save_path = '/home/liuwei/workspace/code/deep_learning_check_v2'

    characters = [x.strip('\n') for x in open('vocab/word/synth_text_vocab', 'r', encoding='utf-8').readlines()]
    characters = ['blank'] + characters + ['end']
    max_length = 25
    lr = 0.0005
    batch_size = 8
    split_step = 100
    print_step = 100
    eval_step = 10000
    epoch = 100

    os.makedirs(save_path, exist_ok=True)

    dataset = LmdbDataset(data_root)
    filters = BuildFilter(
        data_root,
        [
            LmdbImageFilter(),
            LmdbOutVocabFilter(characters, max_length=max_length)
        ],
        recache=False
    )

    transforms = BuildTransform([
        LmdbImageDecode(),
        ImagePilToNP(),
        ImageResizeNormal((3, 32, 256)),

        LmdbStreamDecode(),
        TextEncode(characters),

    ])

    build_dataset = BuildDataset(dataset, filters, transforms)

    collate = BuildTransform([
        BatchSplit(),

        ImageCollate(),
        LabelCollate(max_length=max_length),

        NPToTensor()
    ])
    dataloader = MutilDataLoader(
        [build_dataset],
        batch_size,
        collate
    )

    modules = [
        ResNet(3, 512),
        nn.AdaptiveAvgPool2d((1, -1)),
        EncoderWithRNN(512, 512 // 2),
        CTC(512, len(characters))
    ]
    model = CRNN(3, 512, 512, len(characters), save_path)
    model.to(device)
    model.resume_model(os.path.join(save_path, 'latest.pth'))

    base_model = CRNN(3, 512, 512, len(characters), save_path)
    base_model.to(device1)
    base_model.resume_model(os.path.join(save_path, 'latest.pth'))

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99),
    #                              weight_decay=0.00004)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                 weight_decay=0.00004)
    base_lr = [group['lr'] for group in optimizer.param_groups]
    scheduler = Warmup(base_lr=base_lr, warm=10000)
    optimizer_with_scheduler = OptimizerWithScheduler(optimizer=optimizer, scheduler=scheduler)

    criterion = RLHFLossProxy(0.5)

    metric = RecMetric()
    decoder = TextDecoder(characters)

    index = 0
    loss_v = 0
    for i in range(epoch):
        bar = tqdm(dataloader)
        for item in bar:
            # print(item)
            images, labels = item
            texts, lengths = labels
            images = images.to(device)
            texts = texts.to(device)

            preds = model(images)
            with torch.no_grad():
                base_pred = base_model(images.to(device1)).to(device)

            loss, radio = criterion(preds, base_pred)

            optimizer_with_scheduler.zero_grad()
            loss.backward()
            curr_lr = optimizer_with_scheduler.step()
            _, pred_indices = preds.max(dim=2)
            preds, labels = decoder(pred_indices), decoder(texts)
            metric(preds, labels)

            loss_v += radio
            index += 1
            if index % print_step == 0:
                curr_lr = np.round(curr_lr, 7)
                radio = np.round(float(loss_v) / print_step, 5)
                acc = metric.get_metric()
                bar.set_postfix_str(f'{index},{radio},{acc}')
                metric.reset()
                loss_v = 0
            if index % eval_step == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'latest_v1.pth'))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
