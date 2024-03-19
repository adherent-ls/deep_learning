import torch

from base.module.base_loss import BaseLoss


class RLHFLossProxy(BaseLoss):
    def __init__(self, thresh, acc=0.8):
        super(RLHFLossProxy, self).__init__()
        self.good_thresh = 0.983
        self.bad_thresh = thresh / 2
        self.acc = acc

    def loss_fn_v1(self, pred, base_pred):
        cos = torch.cosine_similarity(pred, base_pred, dim=2)  # b,n,c -> b,n
        cos = cos.mean(dim=1)  # b,n -> b
        pred_good = pred * (cos > self.good_thresh)[:, None, None] * -1
        pred_bad = pred * (cos <= self.bad_thresh)[:, None, None] * 1
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
        pred_bad = pred * (same[:, :, None] <= self.bad_thresh) * 1
        loss = (pred_good + pred_bad).mean(0).sum()
        return loss, 0

    def loss_fn_v3(self, pred, base_pred):
        cos = torch.cosine_similarity(pred, base_pred, dim=1)  # b,n,c -> b,n
        cos = cos.mean(dim=1)  # b,n -> b

        # pred = torch.abs(pred)
        pred_best = pred * (cos > self.good_thresh)[:, None, None] * 0

        pred_g = pred * (cos <= self.good_thresh)[:, None, None] * 0.5
        pred_bad = pred * (cos <= self.bad_thresh)[:, None, None] * 1
        pred_t = pred * (cos <= 0)[:, None, None] * 2
        loss = (pred_best + pred_g + pred_bad + pred_t).mean(dim=0).sum()

        return loss, cos.mean()

    def loss_fn_v4(self, pred, base_pred):
        base_pred_a, base_indices = base_pred.detach().max(dim=-1)
        pred_a, indices = pred.max(dim=-1)
        same = (indices == base_indices).sum(dim=1) / pred.shape[1]

        pred_good = pred_a * (same >= same.median())[:, None] * -1 * same[:, None]
        pred_bad = pred_a * (same <= same.median())[:, None] * 1 * (1 - same[:, None])
        loss = (pred_good + pred_bad).mean(dim=0).sum()

        return loss, same.mean()

    def loss_fn_v5(self, pred, base_pred):
        base_pred_a, base_indices = base_pred.detach().max(dim=-1)
        pred_a, indices = pred.max(dim=-1)
        same = (indices == base_indices).float()

        pred_good = pred_a * (same > self.good_thresh)[None] * -0.1
        pred_bad = pred_a * (same <= self.good_thresh)[None] * 0.1
        loss = (pred_good + pred_bad)
        loss = loss / torch.abs(loss).sum()
        loss = loss.sum()
        return loss, same.mean()

    def __call__(self, pred, base_pred):
        return self.loss_fn_v5(pred, base_pred)
