from utils.register import register


def metric_instance(config):
    return register.get_instance_by_name(config['name'], config['args'], 'metric')


def decoder_instance(config):
    return register.get_instance_by_name(config['name'], config['args'], 'decoder')
