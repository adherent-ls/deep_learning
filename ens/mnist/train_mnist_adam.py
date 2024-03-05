import os.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.modules.backbone.cnn.resnet import resnet18
from utils.cos_sim import batch_cosine_similarity_torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 修改最后的全连接

    def forward(self, x):
        rs, y = self.model(x)
        return rs, y


def main():
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = '/home/data/data/mnist'
    name = 'adam-noclip'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])

    trainset = torchvision.datasets.MNIST(root=root, train=True,
                                          download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=root, train=False,
                                         download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

    # 定义ResNet50模型
    model = Model().model
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.00004)

    step = 0
    best = 0
    for i in range(1000):
        for image, label in train_dataloader:
            image = image.to(device)
            label = label.to(device)
            image = torch.cat([image, image, image], dim=1)

            x, pred = model(image)
            loss = criterion(pred, label)
            # for item in model.modules():
            #     l = torch.clip(batch_cosine_similarity_torch(item, item) - 0.5, 0, 1)
            #     l = l.abs().mean(dim=0).mean()
            #     loss += l * 100

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                print(step, loss)
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in valid_dataloader:
                image, label = data
                image = torch.cat([image, image, image], dim=1)
                image = image.to(device)
                label = label.to(device)
                # calculate outputs by running images through the network
                x2, outputs = model(image)
                # the class with the highest energy is what we choose as prediction
                _, predicted = outputs.max(dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        curr = np.round(100 * correct / total, 5)
        print(f'Accuracy of the network on the {len(valid_dataloader)} test images: {curr} %')
        if curr >= best:
            torch.save(model.state_dict(), os.path.join(root, f'best-{name}.pth'))
            best = curr
        torch.save(model.state_dict(), os.path.join(root, f'latest-{name}.pth'))
        print(curr, best, model.fc.weight.var())


if __name__ == '__main__':
    main()
