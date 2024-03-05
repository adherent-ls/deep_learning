import cv2
import numpy as np
import torch.optim
from torch import nn


def batch_cosine_similarity(d1, d2):
    d2 = d2.T
    d1_n, d2_n = np.linalg.norm(d1, axis=-1, keepdims=True), np.linalg.norm(d2, axis=0, keepdims=True)
    c1 = np.matmul(d1, d2) / (d1_n * d2_n)
    return c1


def batch_cosine_similarity_torch(d1, d2):
    d2 = d2.T
    d1_n, d2_n = torch.norm(d1, dim=-1, keepdim=True), torch.norm(d2.T, dim=-1, keepdim=True).T
    c1 = torch.matmul(d1, d2) / (d1_n * d2_n)
    return c1


def single_cosine_similarity(d1, d2):
    d1_n, d2_n = np.linalg.norm(d1), np.linalg.norm(d2)
    c1 = np.dot(d1, d2) / (d1_n * d2_n)
    return c1


def single_cosine_similarity_torch(d1, d2):
    d1_n, d2_n = torch.norm(d1), torch.norm(d2)
    c1 = torch.dot(d1, d2) / (d1_n * d2_n)
    return c1


def construct_tensor(m, n):
    m = m - 1
    if m > 0:
        d0 = construct_tensor(m, n)
        d1 = np.eye(n)
        l = n // (2 ** m)
        idx = np.arange(0 + l, n + l) % n
        d1 = d1[idx]
        d0 = d0.reshape((-1, n, n))
        data = np.concatenate([d0 + d1, d0 * -1 + d1], axis=0)
        data = data.reshape((-1, n))
    else:
        data = np.eye(n)
        data = np.concatenate([data, data * -1], axis=0)
    return data


def main():
    # n = 100
    # m = 2
    # data = construct_tensor(m, n // 2 ** m)
    # data = np.array([
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # ])
    data = np.clip((np.random.random((51200)).reshape((100, 512)) - 0.5) * 2, 0, 1)

    sim = batch_cosine_similarity(data, data)
    n, m = data.shape
    model = nn.Linear(m, n, bias=False)

    # optim = torch.optim.Adam(lr=1, params=model.parameters(), betas=(0.9, 0.9))
    optim = torch.optim.Adam(lr=1, params=model.parameters())

    loss_fn = nn.CrossEntropyLoss()

    ori = data
    for i in range(100000):
        idx = np.random.randint(0, n, (100))
        # idx = np.arange(0, n)
        x = ori[idx]
        x = torch.Tensor(x)
        w1 = model.weight.detach().numpy() + 0

        y = model(x)
        loss = loss_fn(y, torch.Tensor(idx).long())
        loss.backward()
        # loss = torch.clip((batch_cosine_similarity_torch(x, model.weight) + 1) / 2, 0, 1)
        # loss = -1 * torch.log(loss)
        # loss = loss[idx, idx].sum()
        # loss.backward()
        optim.step()
        optim.zero_grad()
        weight = model.weight.detach().numpy()
        print(weight.sum(), weight.mean(), loss)
        if loss < 0.01 and i > 0:
            break
    x = torch.Tensor(ori)
    w_sim = batch_cosine_similarity(weight, weight)
    sim_s = sim * w_sim
    sim_s = sim_s - np.eye(sim_s.shape[0])

    weight_pos = (weight > 0) * weight
    weight_neg = (weight < 0) * weight

    pred_pos = x @ weight_pos.T
    pred_neg = x @ weight_neg.T

    pos_r = (torch.max(pred_pos, dim=1)[1] == torch.arange(0, n)).sum()
    neg_r = (torch.max(pred_neg, dim=1)[1] == torch.arange(0, n)).sum()
    print(pos_r, neg_r)

    step_l = 0.1
    for j in range(n):
        x1 = ori[j:j + 1]
        x1 = torch.Tensor(x1)
        g = 0
        loss = 0
        for i in range(1000):
            x2 = x1 + g
            x2.requires_grad = True
            y1 = model(x2)
            # y1 = x2 @ torch.Tensor(weight_neg).T
            loss = loss_fn(y1, torch.Tensor([j]).long())
            loss.backward()
            # y1 = batch_cosine_similarity_torch(x2, model.weight)
            # loss = torch.clip((y1 + 1) / 2, 0, 1)
            # loss = -1 * torch.log(loss)
            # loss = loss[[0], [j]].sum()
            # loss.backward()

            grad = torch.sign(x2.grad) * step_l * 0.1
            # grad = torch.sign(model.weight[j].detach()) * -step_l * 0.1
            g = g + grad + torch.rand(grad.shape) * step_l * 0.01
            g = torch.clip(g, -step_l, step_l)
        pred = y1[0].detach().numpy().argmax()
        if pred != j:
            t = 0
        print(loss, j, pred)
    t = 0


if __name__ == '__main__':
    main()
