import yaml


def get_param(item, param):
    return eval(item[1:])


def get_config(item, config):
    return eval(item[1:])


def load(item, k, param):
    config = yaml.load(open(item[7:-2], 'rb'), Loader=yaml.Loader)
    config = build_item(config, None, config, param)
    if k is not None and k in config:
        config = config[k]
    return config


def build_item(item, key, config, param):
    if isinstance(item, str) and item.startswith('$load'):
        result = load(item, key, param)
    elif isinstance(item, str) and item.startswith('$param'):
        result = get_param(item, param)
    elif isinstance(item, str) and item.startswith('$config'):
        result = get_config(item, config)
    elif isinstance(item, str) and item.startswith('$'):
        result = eval(item[1:])
    elif isinstance(item, list):
        for i, children in enumerate(item):
            item[i] = build_item(children, None, config, param)
        result = item
    elif isinstance(item, dict):
        for k, v in item.items():
            vs = build_item(v, k, config, param)
            item[k] = vs
        result = item
    else:
        result = item
    return result


def build_str_compair_config(config):
    config = build_item(config, None, config, config)
    return config


def build_config(path):
    from utils.yaml_loader import YamlLoader
    config = yaml.load(open(path, 'rb'), YamlLoader)
    # yaml.dump(config, open('config.yaml', 'w', encoding='utf-8'), Dumper=yaml.Dumper)
    return config
