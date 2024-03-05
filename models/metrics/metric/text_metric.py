import Levenshtein

from base.metric.base_metric import BaseMetric


class TextMetric(BaseMetric):
    def __init__(self):
        super(TextMetric, self).__init__()
        self.acc = 0
        self.nol = 0
        self.t = 0

    def reset(self):
        self.acc = 0
        self.nol = 0
        self.t = 0

    def __call__(self, pred_texts, labels):
        for text, label in zip(pred_texts, labels):
            if text == label:
                self.acc += 1
            self.nol += 1 - Levenshtein.distance(text, label) / max(len(label), len(text))
            self.t += 1
        return self.nol / self.t

    def once(self, pred_texts, labels):
        acc = 0
        nol = 0
        t = 0
        for text, label in zip(pred_texts, labels):
            if text == label:
                acc += 1
            nol += 1 - Levenshtein.normalized_distance(text, label) / max(len(label), len(text))
            t += 1
        return nol / t

    def value(self):
        return self.nol / self.t
