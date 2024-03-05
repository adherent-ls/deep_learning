import time

import numpy as np
import torch
from torch.nn import functional as F

module_output = {}


def module_hook(module, input, output):
    if len(list(module.children())) > 0:
        pass
    elif hasattr(output, 'grad_fn'):
        func = output.grad_fn
        module_output[func] = [module, input, output]
    elif type(output) is tuple:
        for item in output:
            if hasattr(item, 'grad_fn') and item.grad_fn is not None:
                func = item.grad_fn
                module_output[func] = [module, input, output]
    else:
        print()


def build_graph(tensor, graph_dict):
    if tensor is None:
        return graph_dict
    next = tensor.next_functions
    graph_dict[tensor] = [item[0] for item in next]
    for item in next:
        build_graph(item[0], graph_dict)
    return graph_dict


def build_visual(graph_dict, module_output, start, weight, type):
    next = start
    curr = graph_dict[next]
    weight_r = 0, 0
    for item in curr:
        b = 0
        name = item.__class__.__name__
        if name == 'AddBackward0':
            weight_item = build_visual(graph_dict, module_output, item, weight, type)
            weight_r = weight_r[0] + weight_item[0], weight_r[1] + weight_item[1]
            continue
        elif name == 'ReluBackward1' and (type == 'relu' or type == 'maxpool'):
            weight_item = relu_maxpool_backward(module_output[item][2][0], weight)
            new_type = 'relu'
        elif name == 'ReluBackward1' and type == 'conv':
            weight_item = relu_conv_backward(module_output[item][2][0], weight)
            new_type = 'conv'
        elif name == 'ReluBackward0' and (type == 'relu' or type == 'maxpool'):
            weight_item = relu_maxpool_backward(module_output[item][2][0], weight)
            new_type = 'relu'
        elif name == 'ReluBackward0' and type == 'conv':
            weight_item = relu_conv_backward(module_output[item][2][0], weight)
            new_type = 'conv'
        elif name == 'CudnnBatchNormBackward' and type == 'relu':
            weight_item, b = bn_relu_backward(module_output[item][0].running_mean,
                                              module_output[item][0].running_var,
                                              module_output[item][0].weight,
                                              module_output[item][0].bias,
                                              module_output[item][0].eps,
                                              weight)
            new_type = 'relu'
            # weight_item, b = weight[0], weight[1]
        elif name == 'CudnnBatchNormBackward' and type == 'conv':
            weight_item, b = bn_conv_backward(module_output[item][0].running_mean,
                                              module_output[item][0].running_var,
                                              module_output[item][0].weight,
                                              module_output[item][0].bias,
                                              module_output[item][0].eps,
                                              weight)
            new_type = 'conv'
            # weight_item, b = weight[0], weight[1]
        elif name == 'CudnnConvolutionBackward' and type == 'relu':
            weight_item = conv_relu_backward(module_output[item][0].weight,
                                             module_output[item][0].stride,
                                             module_output[item][0].padding,
                                             weight)
            new_type = 'conv'
        elif name == 'CudnnConvolutionBackward' and type == 'conv':
            weight_item = conv_conv_backward(module_output[item][0].weight,
                                             module_output[item][0].stride,
                                             module_output[item][0].padding,
                                             weight)
            new_type = 'conv'
        elif name == 'ThnnConv2DBackward' and type == 'relu':
            weight_item = conv_relu_backward(module_output[item][0].weight,
                                             module_output[item][0].stride,
                                             module_output[item][0].padding,
                                             weight)
            new_type = 'conv'
        elif name == 'ThnnConv2DBackward' and type == 'conv':
            weight_item = conv_conv_backward(module_output[item][0].weight,
                                             module_output[item][0].stride,
                                             module_output[item][0].padding,
                                             weight)
            new_type = 'conv'
        elif name == 'MaxPool2DWithIndicesBackward' and type == 'conv':
            weight_item = maxpool_conv_backward(module_output[item][2][1][0],
                                                module_output[item][0].kernel_size,
                                                module_output[item][0].stride,
                                                module_output[item][0].padding,
                                                module_output[item][1][0].shape,
                                                weight)
            new_type = 'conv'
        elif name == 'MaxPool2DWithIndicesBackward' and type == 'relu':
            weight_item = maxpool_relu_backward(module_output[item][2][1][0],
                                                module_output[item][0].kernel_size,
                                                module_output[item][0].stride,
                                                module_output[item][0].padding,
                                                weight)
            new_type = 'relu'
        elif name == 'AccumulateGrad':
            continue
        elif name == 'NoneType':
            continue
        else:
            raise ModuleNotFoundError
        weight_item, b1 = build_visual(graph_dict, module_output, item, weight_item, new_type)
        weight_r = (weight_r[0] + weight_item, weight_r[1] + b + b1)
    if not weight_r[0].__class__ is torch.Tensor:
        weight_r = weight, 0
    return weight_r


def conv_conv_combine_backward(weight1, weight2):
    ou1, in1, h1, w1 = weight1.shape
    ou2, in2, h2, w2 = weight2.shape
    if h1 == h2 and w1 == w2:
        out = weight1 + weight2
    else:
        out = torch.zeros((ou2, in1, max(h1, h2), max(w1, w2)), device=weight1.device)
        out[:, :, h1 // 2: h1 // 2 + 1, w1 // 2: w1 // 2 + 1] = weight1
        out[:, :, h2 // 2: h2 // 2 + 1, w2 // 2: w2 // 2 + 1] = weight2
    return out


def conv_conv_backward(weight1, s1, p1, weight2):
    ou1, in1, h1, w1 = weight1.shape
    ou2, in2, h2, w2 = weight2.shape

    p1x, p1y = p1
    s1x, s1y = s1
    h, w = h2 * s1x + h1 - 1, w2 * s1y + w1 - 1
    out = torch.zeros((ou2, in1, h2 * s1x + h1 - 1, w2 * s1y + w1 - 1), device=weight1.device)
    with torch.no_grad():
        for k in range(0, h2):
            for m in range(0, w2):
                item_v = (weight2[:, :, k, m] @ weight1.reshape(ou1, -1)).reshape(ou2, in1, h1, w1)
                out[:, :, k * s1x:k * s1x + h1, m * s1y:m * s1y + w1] += item_v
    out = out[:, :, p1x:h - p1x, p1y:w - p1y]
    # TODO 需要的显存过大
    # we1 = weight1.reshape(ou1, in1, h1 * w1)[None, :, :, :, None]
    # we2 = weight2.reshape(ou2, in2, h2 * w2)[:, :, None, None, :]
    # we = torch.sum(we1 * we2, dim=1).reshape(ou2, in1 * h1 * w1, h2 * w2)
    # out2 = F.fold(we, (5, 5), kernel_size=(3, 3), stride=s1, padding=0)
    # out2 = out2[:, :, p1x:, p1y:]
    return out


def conv_maxpool_backward(weight, s1, p1, pool):
    oup, hp, wp = pool.shape
    ouc, inc, hc, wc = weight.shape
    p1x, p1y = p1
    s1x, s1y = s1

    h, w = hp * s1x + hc - 1, wp * s1y + wc - 1
    out = torch.zeros((ouc, inc, h, w), device=weight.device)
    for i in range(0, hp):
        for j in range(0, wp):
            pos = pool[:, i, j]
            out[:, :, i * s1x:i * s1x + hc, j * s1y:j * s1y + wc] += weight[:, :] * pos[:, None, None, None]
    out = out[:, :, p1x:h - p1x, p1y:w - p1y]
    return out


def bn_relu_backward(miu, sigma, gamma, beta, eps, rv):
    vx = torch.sqrt(sigma[:, None, None] + eps)

    rv_p = rv > 0
    out = rv_p * gamma[:, None, None] / vx

    nb = miu * gamma / vx[:, 0, 0]
    # print(weight.sum(dim=[1, 2, 3]))
    b = ((-nb + beta) * rv_p.sum(dim=[1, 2]))
    return out, b


def bn_conv_backward(miu, sigma, gamma, beta, eps, weight):
    vx = torch.sqrt(sigma[None, :, None, None] + eps)
    out = weight * gamma[None, :, None, None] / vx

    nb = miu[None, :] * gamma[None, :] / vx[:, :, 0, 0]
    beta = beta[None, :]
    # print(weight.sum(dim=[1, 2, 3]))
    return out, ((-nb + beta) * weight.sum(dim=[2, 3])).sum(dim=[1])


def conv_bn_backward(weight, s1, p1, miu, sigma, shape):
    oup, hp, wp = shape
    ouc, inc, hc, wc = weight.shape
    p1x, p1y = p1
    s1x, s1y = s1

    h, w = hp * s1x + hc - 1, wp * s1y + wc - 1

    out = torch.zeros((ouc, inc, h, w), device=weight.device)
    for k in range(oup):
        for i in range(0, hp):
            for j in range(0, wp):
                pos = sigma[k]
                out[k, :, i * s1x:i * s1x + hc, j * s1y:j * s1y + wc] += weight[k, :] / pos
    out = out[:, :, p1x:h - p1x, p1y:w - p1y]

    out1 = torch.zeros((ouc, inc, hp * s1x + hc - 1, wp * s1y + wc - 1), device=weight.device)
    for i in range(0, hp):
        for j in range(0, wp):
            pos = sigma
            out1[:, :, i * s1x:i * s1x + hc, j * s1y:j * s1y + wc] += weight[:, :] / pos
    out1 = out1[:, :, p1x:h - p1x, p1y:w - p1y]
    return out, -1 * miu / sigma


def conv_relu_backward(weight, s1, p1, rv):
    oup, hp, wp = rv.shape
    ouc, inc, hc, wc = weight.shape
    p1x, p1y = p1
    s1x, s1y = s1

    h, w = hp * s1x + hc - 1, wp * s1y + wc - 1
    out = torch.zeros((ouc, inc, h, w), device=weight.device)
    for i in range(0, hp):
        for j in range(0, wp):
            pos = rv[:, i, j]
            out[:, :, i * s1x:i * s1x + hc, j * s1y:j * s1y + wc] += weight[:, :] * pos[:, None, None, None]
    out = out[:, :, p1x:h - p1x, p1y:w - p1y]
    return out


def maxpool_conv_backward(pool, kernel_size, stride, padding, input_shape, weight):
    inc, hp, wp = pool.shape
    ouc, inc, hc, wc = weight.shape
    _, _, sx, sy = input_shape
    out = torch.zeros((ouc, inc, sx, sy), device=weight.device)
    if kernel_size > stride:
        # 在可能存在重复选择同一个项目时，应该每一个位置单独处理
        # assert hc == hp and wc == wp

        for i in range(hp):
            for j in range(wp):
                pos = pool[:, i, j]
                x, y = pos // sx, pos % sy
                out[:, torch.arange(inc), x, y] += weight[:, :, i, j]
    else:
        for i in range(hp - hc + 1):
            for j in range(wp - wc + 1):
                pos = pool[:, i:i + hc, j:j + wc]
                x, y = pos // sx, pos % sy
                out[:, torch.arange(inc)[:, None, None], x, y] += weight[:, :]
    # out1 = F.max_unpool2d(weight, pool[None].repeat(weight.shape[0], 1, 1, 1), kernel_size, stride, padding,
    #                      output_size=(hp * stride, wp * stride))
    return out


def relu_maxpool_backward(rv, pool):
    rv_p = rv > 0
    # rv_p = torch.ones_like(rv)
    return rv_p * pool


def relu_conv_backward(rv, weight):
    inc, hr, wr = rv.shape
    ouc, inc, hc, wc = weight.shape

    rv_p = rv > 0
    # rv_p = torch.ones_like(rv)

    out = torch.zeros((ouc, inc, hr, wr), device=weight.device)
    if hr >= hc and wr >= wc:
        for i in range(hr - hc + 1):
            for j in range(wr - wc + 1):
                out[:, :, i:i + hc, j:j + wc] += rv_p[None, :, i:i + hc, j:j + wc] * weight[:, :]
    else:
        out = rv_p[None, :, :, :] * weight[:, :, :hr, :wr]
    return out


def conv_latest(weight, s1, p1):
    # TODO 未能找到在能够表示s的单一卷积核
    ouc, inc, hc, wc = weight.shape
    p1x, p1y = p1
    s1x, s1y = s1

    h, w = hc * s1x, wc * s1y
    out = torch.zeros((ouc, inc, h, w), device=weight.device)
    for i in range(hc):
        for j in range(wc):
            out[:, :, i * s1x:i * s1x + hc, j * s1y:j * s1y + wc] += weight
    out = out[:, :, p1x:h - p1x, p1y:w - p1y]
    return out


def maxpool_latest(pool, stride):
    inc, hp, wp = pool.shape
    out = torch.zeros((inc, hp * stride, wp * stride), device=pool.device)
    for k in range(inc):
        pos = pool[k]
        x, y = pos // (stride * wp), pos % (stride * wp)
        out[k, x, y] = 1
    return out


def relu_latest(rv):
    out = rv > 0
    # out = torch.ones_like(rv)
    return out


def main():
    d = torch.ones((1, 3, 28, 28))
    from torch import nn

    class LocalModel(nn.Module):
        def __init__(self, ic=256, oc=512):
            super().__init__()
            self.conv1 = nn.Conv2d(ic, oc, 1, bias=False, stride=2, padding=0)
            self.conv2 = nn.Conv2d(oc, oc, 3, bias=False, stride=1, padding=1)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm2d(oc)
            self.bn1.running_mean = torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            self.bn1.running_var = 1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            self.bn1.weight = nn.Parameter(1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)
            self.bn1.bias = nn.Parameter(torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)

            self.bn2 = nn.BatchNorm2d(oc)
            self.bn2.running_mean = torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            self.bn2.running_var = 1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            self.bn2.weight = nn.Parameter(1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)
            self.bn2.bias = nn.Parameter(torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)

            bn2 = nn.BatchNorm2d(oc)
            bn2.running_mean = torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            bn2.running_var = 1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1
            bn2.weight = nn.Parameter(1 + torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)
            bn2.bias = nn.Parameter(torch.clip(torch.rand((oc,)) * 0.2, 0, 0.2) - 0.1)
            self.downsample = nn.Sequential(
                nn.Conv2d(ic, oc, 1, bias=False, stride=2, padding=0),
                bn2,
            )

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    from ens.mnist.train_mnist import Model
    model = Model().model.cuda()
    # for item in model.modules():
    #     if type(item) is torch.nn.BatchNorm2d:
    #         item.bias = torch.nn.Parameter(torch.zeros(item.bias.shape, device=item.bias.device))
    #         item.weight = torch.nn.Parameter(torch.ones(item.weight.shape, device=item.bias.device))
    #         item.running_mean = torch.zeros(item.running_mean.shape, device=item.bias.device)
    # item.running_var = torch.ones(item.running_var.shape, device=item.bias.device)
    # model = LocalModel().cuda()
    model.eval()

    for item in model.modules():
        item.register_forward_hook(module_hook)

    d = d.cuda()
    output = model(d)
    graph_dict = build_graph(output.grad_fn, {})
    start = output.grad_fn
    # weight = maxpool_latest(module_output[start][2][1][0],
    #                         module_output[start][0].stride)
    weight = relu_latest(output[0])
    weight = build_visual(graph_dict, module_output, start, weight, 'relu')

    # pool = relu_latest(r2v[0])
    # conv_w = conv_relu_backward(conv2.weight, conv2.stride, conv2.padding, pool)
    # conv_w = relu_conv_backward(r1v[0], conv_w)
    # # conv_w, b = bn_conv_backward(bn.running_mean, bn.running_var, conv_w)
    # conv_w = conv_conv_backward(conv1.weight, conv1.stride, conv1.padding, conv_w)
    # conv_w = maxpool_conv_backward(pos[0], pooling.kernel_size, pooling.stride, pooling.padding, conv_w)
    r2 = F.conv2d(d.cuda(), weight[0], bias=weight[1])
    print()

    t = F.unfold(d, (3, 3))
    print(t)


def test():
    model = torch.nn.Conv2d(1, 1, 3, 1, 1)


if __name__ == '__main__':
    main()
