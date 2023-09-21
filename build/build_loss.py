from utils.register import register


class MultiDictLoss(object):
    def __init__(self, config):
        self.loss = self.build_loss_fn(config['Loss'])
        self.keys = config['keys']
        if 'weight' not in config:
            self.weight = [1] * len(self.keys)
        else:
            self.weight = config['weight']

    def build_loss_fn(self, config):
        losses = []
        for item in config:
            k = item['name']
            v = item['args']
            loss_item = register.build_from_config(k, v, 'loss')
            losses.append(loss_item)
        return losses

    def __call__(self, predication, target):
        loss_v = 0
        for item, keys, weight in zip(self.loss, self.keys, self.weight):
            predication_keys = keys[0]
            target_keys = keys[1]

            if isinstance(predication_keys, str):
                predication_keys = [predication_keys]
            if isinstance(target_keys, str):
                target_keys = [target_keys]

            pred, target_item = [], []
            for key in predication_keys:
                pred.append(predication[key])
            for key in target_keys:
                target_item.append(target[key])

            if len(predication_keys) > 1:
                pred = [pred]
            if len(target_keys) > 1:
                target_item = [target_item]

            loss_v += weight * item(*pred, *target_item)
        return loss_v
