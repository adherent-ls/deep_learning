from utils.register import register


class MultiDictDecoder(object):
    def __init__(self, config):
        self.decoder = self.build_decoder(config)

    def build_decoder(self, config):
        decoders = []
        for item in config:
            decoder_name = item['name']
            decoder_args = item['args']
            decoders.append(register.build_from_config(decoder_name, decoder_args, 'decoder'))
        return decoders

    def __call__(self, index):
        res = index
        for item in self.decoder:
            res = item(res)
        return res


class MultiDictMetric(object):
    def __init__(self, config):
        self.posts = self.build_metric(config['Metric'])
        self.keys = config['keys']
        if 'weight' not in config:
            self.weight = [1] * len(self.keys)
        else:
            self.weight = config['weight']

    def build_metric(self, config):
        posts = []
        for item in config:
            name = item['name']
            args = item['args']
            if 'decoder' in args:
                args['decoder'] = MultiDictDecoder(args['decoder'])
            posts.append(register.build_from_config(name, args, 'metric'))
        return posts

    def __call__(self, predication, target):
        score = 0
        for post, keys, weight in zip(self.posts, self.keys, self.weight):
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

            score += weight * post(*pred, *target_item)
        return score

    def once(self, predication, target):
        score = 0
        for post, keys, weight in zip(self.posts, self.keys, self.weight):
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

            score += weight * post.once(*pred, *target_item)
        return score

    def reset(self):
        for post in self.posts:
            post.reset()

    def value(self):
        score = 0
        for post, weight in zip(self.posts, self.weight):
            score += post.value() * weight
        return score
