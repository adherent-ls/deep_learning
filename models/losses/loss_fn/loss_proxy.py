import torch
from torch.nn import CrossEntropyLoss, CTCLoss


class CrossEntropyLossProxy(object):
    def __init__(self, ignore_index):
        super(CrossEntropyLossProxy, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, pred_ori, target_ori):
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)
        return self.ce(pred, target)


class CTCLossProxy(object):
    def __init__(self, zero_infinity=False, use_focal_loss=False):
        super(CTCLossProxy, self).__init__()
        self.ctc_fn = CTCLoss(blank=0, zero_infinity=zero_infinity, reduction='none')
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

        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)
        loss = loss.mean()
        return loss


class TotalVariationCrossEntropyLossProxy(object):
    def __init__(self, ignore_index, gama):
        super(TotalVariationCrossEntropyLossProxy, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.gama = gama

    def __call__(self, pred_ori, target_ori):
        target_ori = target_ori.to(pred_ori.device)
        pred = pred_ori.view(-1, pred_ori.shape[-1])

        pred = self.gama + (1 - self.gama) * pred
        target = target_ori.view(-1)
        return self.ce(pred, target) / (1 - self.gama + 1e-8)


class TotalVariationMarinCrossEntropyLossProxy(object):
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


class FocalLossProxy(object):
    def __init__(self, gamma=0.5, ignore_index=-100, reduction='mean'):
        super(FocalLossProxy, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred_ori, target_ori):
        target_ori = target_ori.to(pred_ori.device)
        pred = pred_ori.view(-1, pred_ori.shape[-1])
        target = target_ori.view(-1)

        pred = pred.softmax(dim=1)
        loss = self.ce(pred, target)

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
