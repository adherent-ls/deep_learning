from utils.register import register


def criterion_instance(config):
    return register.get_instance_by_name(config['name'], config['args'], 'loss')
