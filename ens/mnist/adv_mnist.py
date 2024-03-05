import os

import torch
import torchvision
from torch import nn
from torchvision import transforms

from ens.mnist.train_mnist import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def adv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = '/home/data/data/mnist'
    model = Model().model
    model.load_state_dict(torch.load(os.path.join(root, 'best-adam.pth')))

    batch_size = 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root=root, train=False,
                                         download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()

    adv_n = 0
    t = 0
    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        image = torch.cat([image, image, image], dim=1)
        label = label.to(device)
        noise = 0

        x2, pred = model(image)
        # pred_pos = linear_pos(x2)
        # pred_neg = linear_neg(x2)
        # pre_d = [pred_neg, pred_pos, pred]
        pre_d = pred + 0
        for i in range(100):
            image_r = image + noise
            image_r.requires_grad = True
            x2, pred_t = model(image_r)
            loss = criterion(pred_t, pre_d.max(dim=-1)[1])
            loss.backward()

            if pre_d is not None and pre_d.max(dim=-1)[1] != pred_t.max(dim=-1)[1]:
                adv_n += 1
                break
            grad = image_r.grad
            noise += torch.sign(grad) * 0.01
            noise = torch.clip(noise, -0.15, 0.15)
        t += 1
        print(adv_n, t, adv_n / t, noise.abs().mean(), loss)


if __name__ == '__main__':
    adv()
