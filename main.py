from utils.runner import Runner
from config_test import build_config

if __name__ == '__main__':
    config = build_config('configs/text_recognizer/base_update.yaml')
    runner = Runner(config)
    runner.train()
    print(runner)
