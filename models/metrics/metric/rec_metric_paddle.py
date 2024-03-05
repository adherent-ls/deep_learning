import string

import numpy as np
from rapidfuzz.distance import Levenshtein

from base.metric.base_metric import BaseMetric


class RecMetric(BaseMetric):
    def __init__(self,
                 main_indicator='norm_edit_dis',
                 is_filter=False,
                 ignore_space=True):
        super().__init__()
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, preds, labels):
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for pred, target in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        metric = {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
        }
        return np.round(metric[self.main_indicator], 5)

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        metric = {'acc': np.round(acc, 5), 'norm_edit_dis': np.round(norm_edit_dis, 5)}
        return metric[self.main_indicator]

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
