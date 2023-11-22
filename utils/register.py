import importlib
import os
import traceback

from base.data.base_dataset import BaseDataset
from base.data.base_filter import BaseFilter
from base.data.base_transformer import BaseTransformer

from base.model.base_model import BaseModel

from base.optimizer.base_optimizer import BaseOptimizer
from base.optimizer.base_scheduler import BaseScheduler

from base.loss.base_loss import BaseLoss

from base.metric.base_decoder import BaseDecoder
from base.metric.base_post import BaseMetric


class Register(object):
    def __init__(self, root, exact=(), need_register=None):
        if need_register is None:
            self.need_register = {
                # data
                'transform': BaseTransformer,
                'filter': BaseFilter,
                'data': BaseDataset.__base__,
                # model
                'model': BaseModel.__base__,
                # optim
                'optim': BaseOptimizer.__base__,
                'scheduler': BaseScheduler.__base__,
                # loss
                'loss': BaseLoss.__base__,
                # proces
                'decoder': BaseDecoder,
                'metric': BaseMetric,
            }
        else:
            self.need_register = need_register
        self.register = {}
        for k in self.need_register.keys():
            self.register[k] = {}
        self.load_modules(root, exact)

    def import_module(self, package_name):
        module = importlib.import_module(f'{package_name}'.replace('.py', ''))
        dict_items = module.__dict__
        for k, v in dict_items.items():
            if k.startswith('__'):
                continue
            t = v
            while hasattr(t, '__base__'):
                t = t.__base__
                for group_name, group_flag in self.need_register.items():
                    if t is group_flag:
                        self.register[group_name][k] = v

    def load_modules(self, root, exact=()):
        for item in root:
            package_name = item.strip('./').replace('/', '.').replace('\\', '.')
            if package_name in exact:
                continue
            if os.path.isdir(item):
                dirs = os.listdir(item)
                dirs = [os.path.join(item, dir_item) for dir_item in dirs]
                self.load_modules(dirs, exact)
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
            while hasattr(t, '__base__'):
                t = t.__base__
                group_flag = self.need_register[register_group_name]
                if t is group_flag:
                    right = True
                    break
            if not right:
                raise TypeError(
                    f'Must be {group_flag}, but got{v}')
            self.register[register_group_name][source_name] = v
            return v
        return None

    def build_from_config(self, cls_name, cfg, register_group_name):
        if not isinstance(cfg, dict):
            cfg = {}
        args = cfg.copy()

        cls_name = cls_name
        if isinstance(cls_name, str):
            if cls_name not in self.register[register_group_name]:
                obj_cls = self.realtime_import_module(cls_name, register_group_name)
                if obj_cls is None:
                    raise FileNotFoundError(f'not found {cls_name} in register[{register_group_name}]')
            else:
                obj_cls = self.register[register_group_name][cls_name]
        else:
            raise TypeError(
                f'name must be a str or valid name, but got {type(cls_name)}')
        if hasattr(obj_cls, 'initialization'):
            instance = obj_cls.initialization(obj_cls, **args)
        else:
            instance = obj_cls(**args)
        return instance


# dirs = os.listdir('./')
# dirs = [dir_item for dir_item in dirs if os.path.isdir(dir_item) and not dir_item.startswith('.')]
dirs = ['data', 'losses', 'models', 'optimizers', 'metric']
register = Register(dirs, exact=[])
