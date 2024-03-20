import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from build.criterion import criterion_instance
from build.data import dataloader_instance
from build.metric import metric_instance, decoder_instance
from build.model import module_instance
from build.model.build_data_parallel import BuildDataParallel
from build.optimizer import optim_instance


def build_device(device_config):
    if torch.cuda.is_available():
        devices = []
        for item in enumerate(device_config):
            devices.append(torch.device(item))
            print(f"Using GPU: {torch.cuda.get_device_name(item)}")
    else:
        # 如果没有可用的 GPU 设备，使用 CPU
        devices = torch.device("cpu")
        print("No GPU available, using CPU.")
    return devices


def main():
    yaml_path = r'config/text_recognizer/CRNN-BiLSTM-CTC-V1.yml'
    config = yaml.load(open(yaml_path, 'r', encoding='utf-8'), Loader=yaml.Loader)

    print_step = config['print_step']
    eval_step = config['eval_step']
    device = build_device(config['device'])

    os.makedirs(config['save_path'], exist_ok=True)

    train_dataloader = dataloader_instance(config['data']['Train'])
    valid_dataloader = dataloader_instance(config['data']['Eval'])

    model = module_instance(config['model'])
    model.to(device)

    optimizer_with_scheduler = optim_instance(model, config['optimizer'])

    criterion = criterion_instance(config['criterion'])

    decoder = decoder_instance(config['decoder'])
    metric = metric_instance(config['metric'])

    index = 0
    epoch = config['epoch']
    for i in range(epoch):
        bar = tqdm(train_dataloader)
        for item in bar:
            images, labels = item
            images = images.to(device)

            preds = model(images)

            loss = criterion(preds, labels)
            optimizer_with_scheduler.zero_grad()
            loss.backward()
            curr_lr = optimizer_with_scheduler.step()

            metric(preds, labels, curr_lr)
            index += 1
            if index % print_step == 0:
                print(metric.get_metric())
                metric.reset()
            if index % eval_step == 0:
                model.eval()
                with torch.no_grad():
                    for data in valid_dataloader:
                        images, labels = data
                        preds = model(images)
                        metric(preds, labels)
                model.save_model('latest', metric.get_metric())
                metric.reset()
                model.train()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
