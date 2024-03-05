import os

import torch


class BuildModule(object):
    def __init__(self, modules, save_path):
        super().__init__()

        self.save_path = save_path
        self.modules = torch.nn.ModuleList(modules)

    def __call__(self, data):
        y = data
        for item in self.modules:
            y = item(y)
        return y

    def to(self, device):
        self.modules.to(device)

    def save_model(self, name, metric):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'{name}.pth')
        model_state = self.modules.state_dict()
        information = {
            'metric': metric,
            'model': model_state
        }
        torch.save(information, save_name)
        print(f'save:{name}, {save_path}, {str(metric)}')

    def resume_model(self, resume_path, strict=True):
        if resume_path is None:
            print('resume_path is None')
            return None
        if not os.path.exists(resume_path):
            print('not os.path.exists(resume_path)', resume_path)
            return None
        if resume_path == '':
            print('resume_path == ""')
            return None
        print(resume_path)
        config = torch.load(resume_path, map_location='cpu')

        metric = config['metric']
        model_state = config['model']
        self.modules.load_state_dict(model_state, strict=strict)

        return metric

    def parameters(self):
        return self.modules.parameters()
