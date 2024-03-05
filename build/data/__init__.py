from build.data.build_dataset import BuildDataset
from build.data.build_filter import BuildFilter
from build.data.build_transform import BuildTransform
from data.dataloader.multi_dataloader import MutilDataLoader
from utils.register import register


def dataset_instance(config):
    config['Filter']['filters'] = [
        register.get_instance_by_name(filter['name'], filter['args'], 'filter')
        for filter in config['Filter']['filters']]
    filters = BuildFilter(**config['Filter'])

    config['Transform']['transforms'] = [
        register.get_instance_by_name(transform['name'], transform['args'], 'transform')
        for transform in config['Transform']['transforms']]
    transforms = BuildTransform(**config['Transform'])

    build_datasets = []
    for item in config['Dataset']:
        dataset = register.get_instance_by_name(item['name'], item['args'], 'data')
        build_dataset = BuildDataset(dataset, filters, transforms)
        build_datasets.append(build_dataset)
    return build_datasets


def dataloader_instance(config):
    datasets = dataset_instance(config)
    config['collate'] = [
        register.get_instance_by_name(collate['name'], collate['args'], 'transform')
        for collate in config['collate']]
    collate = BuildTransform(transforms=config['collate'])
    dataloader = MutilDataLoader(
        datasets,
        batch_size=config['batch_size'],
        collate_fn=collate
    )
    return dataloader
