from build.data.build_dataset import BuildDataset
from build.data.build_filter import BuildFilter
from build.data.build_transform import BuildTransform
from build.model.build_module import BuildModule
from data.dataloader.multi_dataloader import MutilDataLoader
from utils.register import register


def module_instance(config):
    config['modules'] = [
        register.get_instance_by_name(module['name'], module['args'], 'model')
        for module in config['modules']
    ]
    model = BuildModule(**config)
    return model
