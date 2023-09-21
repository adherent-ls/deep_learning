import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

from base.loss.base_loss import BaseLoss


class ClipContrastiveLoss(BaseLoss):
    def __init__(self):
        super(ClipContrastiveLoss, self).__init__()

    def forward(self, sim_matrix, mask):
        #         sim_matrix = sim_matrix * mask
        positive = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)
        positive_embed = torch.diag_embed(positive, dim1=-2, dim2=-1)
        negative, _ = (sim_matrix - positive_embed).max(dim=-1)
        l = torch.maximum(negative - positive + 0.2, torch.zeros_like(positive)).sum()
        l = l / torch.sqrt(mask.sum())
        return l


class ClassificationContrastiveLoss(BaseLoss):
    def __init__(self, ignore_index=0):
        super(ClassificationContrastiveLoss, self).__init__()
        self.CrossEntropyLoss = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred_ori, target_ori):
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)
        target2 = torch.flip(target_ori, dims=[0]).view(-1)
        loss = self.CrossEntropyLoss(pred, target)
        loss2 = self.CrossEntropyLoss(pred, target2)
        return torch.maximum(loss2 - loss + 0.2, torch.zeros_like(loss)).sum()


if __name__ == '__main__':
    c = ClassificationContrastiveLoss()
    pred = torch.rand(4, 16, 359)
    target = torch.randint(0, 359, (4, 16))
    l = c(pred, target)
    print(l)
