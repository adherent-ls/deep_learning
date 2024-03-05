import importlib
import os
import traceback
from pathlib import Path

import torch.optim

from base.data.base_dataset import BaseDataset
from base.data.base_filter import BaseFilter
from base.data.base_transform import BaseTransform
from base.metric.base_decoder import BaseDecoder
from base.metric.base_metric import BaseMetric
from base.module.base_model import BaseModel
from base.module.base_scheduler import BaseScheduler


class Register(object):
    def __init__(self, root, exact=(), need_register=None):
        content_path = str(Path(__file__).parent.parent)
        if need_register is None:
            self.need_register = {
                # data
                'transform': BaseTransform,
                'filter': BaseFilter,
                'data': BaseDataset.__base__,
                # model
                'model': BaseModel.__base__,
                # optim
                'optim': torch.optim.Optimizer,
                'scheduler': BaseScheduler,
                # loss
                'loss': torch.nn.modules.loss._Loss,
                # proces
                'metric': BaseMetric,
                'decoder': BaseDecoder,
            }
        else:
            self.need_register = need_register
        self.register = {}
        for k in self.need_register.keys():
            self.register[k] = {}
        self.load_modules(root, content_path, exact)

    def import_module(self, package_name):
        import_file = f'{package_name}'.replace('.py', '')
        module = importlib.import_module(import_file)
        dict_items = module.__dict__
        for k, v in dict_items.items():
            if k.startswith('__'):
                continue
            t = v
            while hasattr(t, '__base__'):
                t = t.__base__
                for group_name, group_flag in self.need_register.items():
                    if t is group_flag:
                        name = f'{import_file}.{k}'
                        if name not in self.register[group_name]:
                            self.register[group_name][name] = []
                        self.register[group_name][name].append(v)
                        if k not in self.register[group_name]:
                            self.register[group_name][k] = []
                        self.register[group_name][k].append(v)

    def load_modules(self, root, content_path, exact=()):
        for item in root:
            package_name = item.strip('./').replace('/', '.').replace('\\', '.')
            if package_name in exact:
                continue
            item_path = os.path.join(content_path, item)
            if os.path.isdir(item_path):
                dirs = os.listdir(item_path)
                dirs = [os.path.join(item, dir_item) for dir_item in dirs]
                self.load_modules(dirs, content_path, exact)
            elif item.endswith('.py'):
                package_name = item.strip('./').replace('/', '.').replace('\\', '.')

                try:
                    self.import_module(package_name)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    print("Fail to initial class [{}] with error: "
                          "{} and stack:\n{}".format(package_name, e, str(stack_info)))

    def realtime_import_module(self, source_name, register_group_name):
        name = str(source_name).split('.')[-1]
        package_name = source_name[:-1 * (len(name) + 1)]
        if len(package_name) == 0:
            return None
        module = importlib.import_module(f'{package_name}'.replace('.py', ''))
        dict_items = module.__dict__
        if name in dict_items:
            v = dict_items[name]
            t = v
            right = False
            group_flag = self.need_register[register_group_name]
            while hasattr(t, '__base__'):
                t = t.__base__
                if t is group_flag:
                    right = True
                    break
            if not right:
                raise TypeError(
                    f'Must be {group_flag}, but got{v}')
            self.register[register_group_name][source_name] = v
            return v
        return None

    def get_instance_by_name(self, cls_name, args, register_group_name):
        cls_name = cls_name
        if isinstance(cls_name, str):
            if cls_name not in self.register[register_group_name]:
                obj_cls = self.realtime_import_module(cls_name, register_group_name)
                if obj_cls is None:
                    raise FileNotFoundError(f'not found {cls_name} in register[{register_group_name}]')
            else:
                obj_cls = self.register[register_group_name][cls_name]
                if len(obj_cls) > 1:
                    raise Exception(f'{cls_name} more than 1 in {register_group_name}, they are:{obj_cls}')
                obj_cls = obj_cls[0]
        else:
            raise TypeError(
                f'name must be a str or valid name, but got {type(cls_name)}')
        if args is None:
            obj_cls = obj_cls()
        else:
            obj_cls = obj_cls(**args)
        return obj_cls


dirs = ['data', 'models']
register = Register(dirs, exact=[])
