import torch
import torchvision
from torch import nn
from torchvision import models, transforms


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 512)  # 修改最后的全连接
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        y = self.model(x)
        r = self.fc(y)
        return y, r


def adv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load('/home/data/data/cifar10/best.pth'))

    batch_size = 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='/home/data/data/cifar10', train=False,
                                           download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()

    adv_n = 0
    t = 0
    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        label = label.to(device)

        noise = 0
        pre_d = None
        for i in range(100):
            image_r = image + noise
            image_r.requires_grad = True
            x2, pred = model(image_r)
            loss = criterion(pred, label)
            loss.backward()

            grad = image_r.grad
            noise += grad
            noise = torch.clip(noise, -0.1, 0.1)

            if pre_d is not None and pre_d != pred.max(dim=-1)[1]:
                adv_n += 1
                break
            pre_d = pred.max(dim=-1)[1]
        t += 1
        print(adv_n, t, adv_n / t)


if __name__ == '__main__':
    adv()
