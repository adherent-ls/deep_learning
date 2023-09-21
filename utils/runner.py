import numpy as np
import torch
from tqdm import tqdm

from build.build_data import MutilDataLoader
from build.build_model import Model
from build.build_optimizer import OptimizerWithScheduler
from build.build_loss import MultiDictLoss
from build.build_metric import MultiDictMetric


class Runner(object):
    def __init__(self, config):
        self.config = config

        self.train_data = MutilDataLoader(config['data']['train'])
        self.val_data = MutilDataLoader(config['data']['val'])

        self.model = Model(config['model'])
        self.optimizer = OptimizerWithScheduler(self.model, config['optim'])
        self.loss = MultiDictLoss(config['loss'])

        self.metric = MultiDictMetric(config['metric'])

        self.best_metric = self.model.resume_model()

    def train(self):
        device = self.config['device']
        show_step = self.config['show_step']
        inference_step = self.config['inference_step']
        grad_clip = self.config['grad_clip']
        best_metric = -1000
        step = 0

        self.model.to(device)
        self.model.train()

        if self.best_metric is not None:
            step = self.best_metric['step']
            best_metric = self.best_metric['value']

        if str(inference_step).endswith('e'):
            inference_step = int(len(self.train_data) * float(inference_step[:-1]))

        print(step, best_metric)
        for epoch in range(int(self.config['epoch'])):
            data_loader = self.train_data
            bar = tqdm(data_loader)
            index = step % len(data_loader)
            for input_data, labels in bar:
                if input_data is None:
                    continue
                for k, v in input_data.items():
                    v = v.to(device)
                    input_data[k] = v
                for k, v in labels.items():
                    v = v.to(device)
                    labels[k] = v
                pred_ori = self.model(input_data)
                loss = self.loss(pred_ori, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                step += 1
                index += 1
                if step % show_step == 0:
                    if self.metric is None:
                        metric = loss
                    else:
                        metric = self.metric.once(pred_ori, labels)
                    loss_v = np.around(float(loss), 5)
                    metric = np.around(float(metric), 5)
                    bar.set_postfix_str(f'step:{step}, loss:{loss_v}, metric:{metric}')
                if step % inference_step == 0:
                    metric = self.inference()
                    self.model.save_model('last', {'step': step, 'value': metric})
                    print('curr', metric)
                    if best_metric is None or metric >= best_metric:
                        self.model.save_model('best', {'step': step, 'value': metric})
                        best_metric = metric
                        print('best', metric)

    def inference(self):
        device = self.config['device']
        bar = tqdm(self.val_data)
        self.model.eval()
        value = 0
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(bar):
                if input_data is None:
                    continue
                for k, v in input_data.items():
                    v = v.to(device)
                    input_data[k] = v
                for k, v in labels.items():
                    v = v.to(device)
                    labels[k] = v

                pred_ori = self.model(input_data)

                if self.metric is None:
                    v = self.loss(pred_ori, labels)
                    value = (value * i + v) / (i + 1)
                else:
                    value = self.metric(pred_ori, labels)
                bar.set_postfix_str(f'{value}')
        if self.metric is not None:
            value = self.metric.value()
            self.metric.reset()
        else:
            value = -1 * value
        self.model.train()
        return value