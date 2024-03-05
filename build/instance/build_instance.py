from typing import List

from utils.register import register
from utils.type import T_Instance


class BuildInstance(object):
    def __init__(self):
        super().__init__()

    def single_instance(self, type_name: T_Instance, group_name):
        class_name, args = type_name
        obj_cls = register.get_instance_by_name(class_name, group_name)
        if type(args) is list:
            instance = obj_cls(*args)
        else:
            instance = obj_cls(**args)
        return instance

    def __call__(self, classes: List[T_Instance], group_name):
        instances = []
        for item in classes:
            class_name, args = item
            obj_cls = register.get_instance_by_name(item[0], group_name)
            if type(args) is list:
                instances.append(obj_cls(*args))
            else:
                instances.append(obj_cls(**args))
        return instances
