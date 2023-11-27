from utils.build_param import build_param


class Base(object):

    @staticmethod
    def initialization(self, **kwargs):
        param = build_param(self, '__init__', kwargs)
        obj = self(**param)

        super_param = build_param(self, 'set', kwargs)
        obj.set(**super_param)
        return obj

    def set(self, **kwargs):
        raise NotImplemented
