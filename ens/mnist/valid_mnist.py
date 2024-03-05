import os

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms

from ens.mnist.train_mnist import Model

relu_output = []


def valid():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = '/home/data/data/mnist'
    model = Model().model
    model.load_state_dict(torch.load(os.path.join(root, 'best-sgd.pth')))

    batch_size = 128
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root=root, train=False,
                                         download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

    weight = model.fc.weight
    linear_pos = nn.Linear(model.fc.in_features, model.fc.out_features, bias=False)
    linear_neg = nn.Linear(model.fc.in_features, model.fc.out_features, bias=False)
    weight_pos = (weight > 0) * weight
    weight_neg = (weight < 0) * weight

    linear_pos.weight = nn.Parameter(weight_pos)
    linear_neg.weight = nn.Parameter(weight_neg)

    model.linear_pos = linear_pos
    model.linear_neg = linear_neg

    model = model.to(device)

    # from visual.gradcam import CamExtractor
    # m = CamExtractor(model, -1)
    model.eval()

    correct = 0
    total = 0
    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        image = torch.cat([image, image, image], dim=1)
        label = label.to(device)
        x2, pred_t = model(image)

        # pred_pos = linear_pos(x2)
        _, predicted = pred_t.max(dim=1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    curr = np.round(100 * correct / total, 5)
    print(f'Accuracy of the network on the 10000 test images: {curr} %')


if __name__ == '__main__':
    valid()
