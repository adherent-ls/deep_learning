import os
from typing import Tuple

import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from utils.misc_functions import save_class_activation_images
from visual.gradcam import GradCam
from ens.utils.visual import *


def visual():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = '/home/data/data/mnist'
    from ens.mnist.train_mnist_adam_norelu import Model

    model = Model().model
    model.load_state_dict(torch.load(os.path.join(root, 'best-adam.pth')))
    model.to(device)

    cam_runner = GradCam(model, target_layer='conv1')

    batch_size = 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root=root, train=False,
                                         download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        image = torch.cat([image, image, image], dim=1)
        label = label.to(device)

        cam_image = cam_runner.generate_cam(image, label)

        ori_image = Image.fromarray(((image * 0.5 + 0.5) * 255)[0, 0].cpu().numpy())
        save_class_activation_images(ori_image, cam_image, 'test_ori')

        cam_image = cam_runner.generate_cam(image, torch.Tensor([4]).long())

        ori_image = Image.fromarray(((image * 0.5 + 0.5) * 255)[0, 0].cpu().numpy())
        save_class_activation_images(ori_image, cam_image, 'test_adv')
        break


def save_class(weight, model, image, number):
    fc_weight = model.fc.weight
    c_weight = (fc_weight @ weight[0].reshape(weight[0].shape[0], -1)).reshape(-1, *weight[0].shape[1:])
    c_b = (fc_weight @ weight[1].reshape(weight[1].shape[0], -1)).reshape(-1, *weight[1].shape[1:])
    # c2 = F.conv2d(image.cuda(), c_weight, bias=c_b)
    # print(torch.abs(c2[:, :, 0, 0] - pred).max(), torch.abs(c2[:, :, 0, 0] - pred).mean())
    c_weight = c_weight.cpu().detach().numpy()
    ori_image = Image.fromarray(((image * 0.5 + 0.5) * 255)[0, 0].cpu().numpy())
    cam = c_weight
    # cam = np.sum(cam, axis=0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    for i, cam_item in enumerate(cam):
        cam_item = cam_item[0]
        save_class_activation_images(ori_image, cam_item, f'test{number}_{i}')
    # print(weight, 1)


def save_conv(weight, image, number):
    ori_image = Image.fromarray(((image * 0.5 + 0.5) * 255)[0, 0].cpu().numpy())
    cam = weight[0].detach().cpu().numpy()
    cam = np.sum(cam, axis=0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam_item = cam[0]
    save_class_activation_images(ori_image, cam_item, f'test{number}')


def conv_visual():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = '/home/data/data/mnist'
    from ens.mnist.train_mnist_adam_norelu import Model

    model = Model().model
    model.load_state_dict(torch.load(os.path.join(root, 'best-adam_norelu.pth')))
    model.to(device)

    model.eval()

    batch_size = 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST(root=root, train=False,
                                         download=True, transform=transform)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)
    for item in model.modules():
        item.register_forward_hook(module_hook)
    number = 0
    for data in valid_dataloader:
        image, label = data
        image = image.to(device)
        image = torch.cat([image, image, image], dim=1)
        label = label.to(device)

        x2, pred = model(image)
        print(torch.topk(pred, k=2))

        graph_dict = build_graph(pred.grad_fn, {})
        start = pred.grad_fn.next_functions[1][0].next_functions[0][0].next_functions[0][0]
        # weight = relu_latest(module_output[start][2][0])
        weight = torch.ones((512, 1, 1), device=x2.device)
        weight = build_visual(graph_dict, module_output, start, weight, 'relu')
        r2 = F.conv2d(image.cuda(), weight[0], bias=weight[1])
        print(torch.abs(r2[:, :, 0, 0] - x2).max(), torch.abs(r2[:, :, 0, 0] - x2).mean())

        save_conv(weight, image, number)
        number += 1
        if number > 5:
            break


if __name__ == '__main__':
    conv_visual()
