import Levenshtein
import torch

from base.metric.base_post import BaseMetric


class TextMetric(BaseMetric):
    def __init__(self, decoder):
        super(TextMetric, self).__init__()
        self.decoder = decoder

        self.acc = 0
        self.normal = 0
        self.t_n = 0
        self.t = 0

    def reset(self):
        self.acc = 0
        self.normal = 0
        self.t_n = 0
        self.t = 0

    def __call__(self, word_predication, word):
        _, index = torch.max(word_predication, dim=-1)
        pred_texts = self.decoder(index=index)
        labels = self.decoder(index=word)
        for text, label in zip(pred_texts, labels):
            if text == label:
                self.acc += 1
            self.normal += Levenshtein.distance(text, label)
            self.t_n += max(len(label), len(text))
            self.t += 1
        return 1 - self.normal / self.t_n

    def once(self, word_predication, word):
        acc = 0
        normal = 0
        t_n = 0
        t = 0
        _, index = torch.max(word_predication, dim=-1)
        pred_texts = self.decoder(index=index)
        labels = self.decoder(index=word)
        for text, label in zip(pred_texts, labels):
            if text == label:
                acc += 1
            normal += Levenshtein.distance(text, label)
            t_n += max(len(label), len(text))
            t += 1
        return 1 - normal / t_n

    def value(self):
        return 1 - self.normal / self.t_n
