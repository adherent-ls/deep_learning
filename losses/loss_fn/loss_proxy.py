import torch
from torch.nn import CrossEntropyLoss, CTCLoss


class CrossEntropyLossProxy(CrossEntropyLoss):
    def __init__(self, ignore_index):
        super(CrossEntropyLossProxy, self).__init__(ignore_index=ignore_index)

    def __call__(self, pred_ori, target_ori):
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)
        return super(CrossEntropyLossProxy, self).__call__(pred, target)


class CTCLossProxy(CTCLoss):
    def __init__(self, zero_infinity):
        super(CTCLossProxy, self).__init__(zero_infinity=zero_infinity)

    def __call__(self, preds, labels):
        text, length = labels
        text = text.to(preds.device)
        length = length.to(preds.device)

        batch_size = text.shape[0]
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.log_softmax(2).permute(1, 0, 2)
        return super(CTCLossProxy, self).__call__(preds, text, preds_size, length)


class TotalVariationCrossEntropyLossProxy(CrossEntropyLoss):
    def __init__(self, ignore_index, gama):
        super(TotalVariationCrossEntropyLossProxy, self).__init__(ignore_index=ignore_index)
        self.gama = gama

    def __call__(self, pred_ori, target_ori):
        target_ori = target_ori.to(pred_ori.device)
        pred = pred_ori.view(-1, pred_ori.shape[-1])

        pred = self.gama + (1 - self.gama) * pred
        target = target_ori.view(-1)
        return super(TotalVariationCrossEntropyLossProxy, self).__call__(pred, target) / (1 - self.gama + 1e-8)


class TotalVariationMarinCrossEntropyLossProxy(CrossEntropyLoss):
    def __init__(self, ignore_index, beta, gama, device):
        super(TotalVariationMarinCrossEntropyLossProxy, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index).to(device)
        self.beta = beta
        self.gama = gama

    def __call__(self, pred_ori, target_ori):
        target_ori = target_ori.to(pred_ori.device)
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)

        pred_d = pred.detach()

        target = max(self.beta, (pred_d / (self.gama + (1 - self.gama) * pred_d))) * target

        loss_v = self.ce(pred, target)
        return loss_v


class FocalLossProxy(CrossEntropyLoss):
    def __init__(self, gamma=0.5, ignore_index=-100, reduction='mean'):
        super(FocalLossProxy, self).__init__(ignore_index=ignore_index, reduction='none')
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred_ori, target_ori):
        target_ori = target_ori.to(pred_ori.device)
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)

        pred = pred.softmax(dim=1)
        loss = super(FocalLossProxy, self).__call__(pred, target)

        l, c = pred_ori.shape

        alpha = (1 - pred_ori[torch.arange(0, l), target_ori]) ** self.gamma  # (b*l,c) + (b*l)

        loss *= alpha
        if self.reduction == 'none':
            pass
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
