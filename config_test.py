import yaml


def build_config(path):
    from utils.yaml_loader import YamlLoader
    config = yaml.load(open(path, 'rb'), YamlLoader)
    yaml.dump(config, open('config.yaml', 'w', encoding='utf-8'), Dumper=yaml.Dumper)
    return config
