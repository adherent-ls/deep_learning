import os

import torch
from torch import nn

from utils.register import register


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.resume_path = config['resume_path']
        self.save_path = config['save_path']
        self.strict = config['strict']
        self.modules = self.build_module(config['Modules'])

        name_maps = {}
        for item in self.modules:
            name = item.__class__.__name__

            if name not in name_maps:
                name_maps[name] = 0
            index = name_maps[name] + 1
            name_maps[name] = index

            full_name = f'{name}_{index}'

            setattr(self, full_name, item)  # 加入attr，model调整到cuda时，只会递归属性中的module

    def build_module(self, model_config):
        module_list = []
        for item_config in model_config:
            name = item_config['name']
            args = item_config['args']
            module_obj = register.build_from_config(name, args, 'model')
            module_list.append(module_obj)
        return module_list

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def resume_model(self):
        resume_path = self.resume_path
        if resume_path is None:
            print('resume_path is None')
            return {'step': 0, 'value': 0}
        if not os.path.exists(resume_path):
            print('not os.path.exists(resume_path)', resume_path)
            return {'step': 0, 'value': 0}
        if resume_path == '':
            print('resume_path == ""')
            return {'step': 0, 'value': 0}
        print(resume_path)
        config = torch.load(resume_path, map_location='cpu')
        state_dict = config['model']
        best_metric = config['metric']
        self.load_state_dict(state_dict, strict=self.strict)
        return best_metric

    def save_model(self, name, metric):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'{name}.pth')
        model_state = self.state_dict()
        information = {
            'metric': metric,
            'model': model_state
        }
        torch.save(information, save_name)
        print(f'save:{name}, {save_path}, {str(metric)}')

    def forward(self, x):
        y = x
        for item in self.modules:
            y = item(y)
        return y
