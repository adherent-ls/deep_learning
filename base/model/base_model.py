import torch
from torch import nn

from base.base.base_dict_call import BaseDictCall
from utils.build_param import build_param


class BaseModel(nn.Module, BaseDictCall):
    pass


class Test(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(128, 128)


def main():
    c = Test.initialization(Test, **{'ink': 'i'})
    print(c)


if __name__ == '__main__':
    main()
