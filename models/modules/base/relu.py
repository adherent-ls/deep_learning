import torch


# 定义计算图
class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.clip(x, 0)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input
