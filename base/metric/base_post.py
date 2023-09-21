import numpy as np


class BaseMetric(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        return args
