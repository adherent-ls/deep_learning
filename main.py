import argparse
import os.path
import random

import numpy as np
import torch

from utils.runner import Runner


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', default='configs/text_recognizer/ResNetAdapt-Transformer-CTC.yaml', help='setting')
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    manualSeed = 3407

    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)

    runner = Runner()
    runner.train()


if __name__ == '__main__':
    main()
