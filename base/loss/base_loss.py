from torch.nn.modules.loss import _Loss


class BaseLoss(_Loss):
    def __init__(self):
        super(BaseLoss, self).__init__()
