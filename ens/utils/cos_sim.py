import numpy as np
import torch


def single_cosine_similarity(d1, d2):
    d1_n, d2_n = np.linalg.norm(d1), np.linalg.norm(d2)
    c1 = np.dot(d1, d2) / (d1_n * d2_n)
    return c1


def batch_cosine_similarity(d1, d2):
    d2 = d2.T
    d1_n, d2_n = np.linalg.norm(d1, axis=-1, keepdims=True), np.linalg.norm(d2, axis=0, keepdims=True)
    c1 = np.matmul(d1, d2) / (d1_n * d2_n)
    return c1


def single_cosine_similarity_torch(d1, d2):
    d1_n, d2_n = torch.norm(d1), torch.norm(d2)
    c1 = torch.dot(d1, d2) / (d1_n * d2_n)
    return c1


def batch_cosine_similarity_torch(d1, d2):
    d2T = torch.transpose(d2, -1, -2)
    d1_n, d2_n = torch.norm(d1, dim=-1, keepdim=True), torch.norm(d2, dim=-1, keepdim=True)
    c1 = (torch.matmul(d1, d2T) + 1e-8) / (d1_n * torch.transpose(d2_n, -1, -2) + 1e-8)
    return c1


def main():
    t = np.array([2, -1])
    t1 = np.array([2, 1])
    s = single_cosine_similarity(t, t1)
    print(s)


if __name__ == '__main__':
    main()
