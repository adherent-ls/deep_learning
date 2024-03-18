import torch

from base.module.base_loss import BaseLoss


class RLHFLossProxy(BaseLoss):
    def __init__(self, thresh):
        super(RLHFLossProxy, self).__init__()
        self.good_thresh = thresh
        self.bad_thresh = thresh / 2

    def loss_fn_v1(self, pred, base_pred):
        cos = torch.cosine_similarity(pred, base_pred, dim=2)  # b,n,c -> b,n
        cos = cos.mean(dim=1)  # b,n -> b
        pred_good = pred * (cos > self.good_thresh)[:, None, None] * -1
        pred_bad = pred * (cos < self.bad_thresh)[:, None, None] * 1
        loss = (pred_good + pred_bad).mean(dim=0).sum()

        # cos_var = cos.var(dim=1)  # b,n -> b
        # loss = (pred * (1 - cos[:, :, None]) * torch.clip(1 - cos_var[:, None, None], 0)).mean(dim=0).sum()

        return loss, cos.mean()

    def loss_fn_v2(self, pred, base_pred):
        base_pred_a = base_pred.detach().argmax(dim=-1)
        pred_a = pred.argmax(dim=-1)
        same = (base_pred_a == pred_a) / pred.shape[1]
        same_index = same.argmax(dim=-1)
        same[:, same_index] = 1

        pred_good = pred * (same[:, :, None] > self.good_thresh) * -5
        pred_bad = pred * (same[:, :, None] < self.bad_thresh) * 1
        loss = (pred_good + pred_bad).mean(0).sum()
        return loss, 0

    def loss_fn_v3(self, pred, base_pred):
        cos = torch.cosine_similarity(pred, base_pred, dim=2)  # b,n,c -> b,n
        cos = cos.mean(dim=1)  # b,n -> b
        pred_best = pred * (cos > self.good_thresh)[:, None, None] * -2

        pred_g = pred * (cos < self.good_thresh)[:, None, None] * -1
        pred_bad = pred * (cos < self.bad_thresh)[:, None, None] * 2
        pred_t = pred * (cos < 0)[:, None, None] * 3
        loss = (pred_best + pred_g + pred_bad + pred_t).mean(dim=0).sum()

        return loss, cos.mean()

    def __call__(self, pred, base_pred):
        return self.loss_fn_v3(pred, base_pred)
