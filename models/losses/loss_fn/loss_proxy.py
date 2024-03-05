import torch
from torch.nn import CrossEntropyLoss, CTCLoss

from base.module.base_loss import BaseLoss


class CrossEntropyLossProxy(BaseLoss):
    def __init__(self, ignore_index):
        super(CrossEntropyLossProxy, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, pred_ori, target_ori):
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)
        return self.ce(pred, target)


class CTCLossProxy(BaseLoss):
    def __init__(self, zero_infinity=False, use_focal_loss=False):
        super(CTCLossProxy, self).__init__()
        self.ctc_fn = CTCLoss(blank=0, zero_infinity=zero_infinity)
        self.use_focal_loss = use_focal_loss

    def __call__(self, preds, labels):
        text, length = labels
        text = text.to(preds.device)
        length = length.to(preds.device)

        batch_size = text.shape[0]
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.log_softmax(2)
        preds = preds.permute(1, 0, 2)
        loss = self.ctc_fn(preds, text, preds_size, length)
        return loss
