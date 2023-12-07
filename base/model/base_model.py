import torch
from torch import nn

from base.base.base_dict_call import BaseDictCall
from utils.build_param import build_param


class BaseModel(nn.Module, BaseDictCall):
    def __call__(self, *args, **kwargs):
        return super(nn.Module, self).__call__(*args, **kwargs)
