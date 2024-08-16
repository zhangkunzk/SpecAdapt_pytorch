#!/usr/bin/env python3
"""
Since: 2022/11/1
Author: Howie
"""
import numpy as np

from .classification import ClassificationMeter

__all__ = [
    'IncrementalMeter'
]


class IncrementalMeter(ClassificationMeter):
    """Class incremental meter
    """
    def __init__(self, num_class):
        super(IncrementalMeter, self).__init__()
        self.num_class = num_class

    def average_accuracy(self, step=10):
        """
        return average
        """
        total_list = [0] * (self.num_class // step)
        predict_list = [0] * (self.num_class // step)
        tp_list = [1e-10] * (self.num_class // step)
        for i, gt in enumerate(self.target_list):
            total_list[gt // step] += 1
            if gt == self.output_list[i]:
                tp_list[gt // step] += 1
            predict_list[self.output_list[i] // step] += 1
        avg_acc = np.array(tp_list) / total_list
        return avg_acc, predict_list, tp_list
