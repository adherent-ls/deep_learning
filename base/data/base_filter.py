from utils.build_param import build_param


class BaseFilter(object):
    def __init__(self):
        pass
    
    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(type(obj), obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj
