#!/usr/bin/env python3

"""
Since: 2022/8/18
Author: Howie
"""

from typing import Tuple

import numpy as np
from sklearn import metrics


class ClassificationMeter(object):

    def __init__(self, num_class):
        self.output_list = []
        self.target_list = []
        self.merged_output = None
        self.merged_target = None
        self.num_class = num_class

    def update(self, output: np.ndarray, target: np.ndarray) -> None:
        """Update the meter.

        Args:
            output: Model output. (n,) vector or (n, 1) matrix.
            target: Ground-truth. (n,) vector or (n, 1) matrix.
        """
        assert len(output.shape) >= 1
        assert len(target.shape) >= 1
        output_flat = output.reshape((-1,))
        target_flat = target.reshape((-1,))
        self.output_list.extend(output_flat)
        self.target_list.extend(target_flat)
        self.merged_output = None
        self.merged_target = None

    def _merge(self):
        if self.merged_target is None:
            self.merged_target = np.array(self.output_list)
            self.merged_output = np.array(self.target_list)

    def accuracy(self) -> float:
        """Compute accuracy score.

        Returns:
            A float number represents the accuracy.
        """
        self._merge()
        return metrics.accuracy_score(self.merged_target, self.merged_output)

    def precision(self) -> np.ndarray:
        """Compute precision scores.

        Returns:
            A (n,) vector, where "n" is the number of classes.
        """
        self._merge()
        return metrics.precision_score(
            self.merged_target,
            self.merged_output,
            average=None
        )

    def recall(self) -> np.ndarray:
        """Compute recall scores.

        Returns:
            A (n,) vector, where "n" is the number of classes.
        """
        self._merge()
        return metrics.recall_score(
            self.merged_target,
            self.merged_output,
            average=None
        )

    def f1(self) -> np.ndarray:
        """Compute f1 scores.

        Returns:
            A (n,) vector, where "n" is the number of classes.
        """
        self._merge()
        return metrics.f1_score(
            self.merged_target,
            self.merged_output,
            average=None
        )

    def precision_recall_f1(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision, recall and f1 scores.

        Returns:
            A tuple of three (n,) vectors represent precision, recall and f1-score,
            where "n" is the number of classes.
        """
        self._merge()
        return metrics.precision_recall_fscore_support(
            self.merged_target,
            self.merged_output,
            average=None
        )[:3]

    def average_accuracy(self, step=1):
        """
        return average
        """
        total_list = [0] * (self.num_class // step)
        pret_list = [0] * (self.num_class // step)
        tp_list = [0] * (self.num_class // step)
        # try:
        for i, gt in enumerate(self.target_list):
            total_list[gt // step] += 1
            if gt == self.output_list[i]:
                tp_list[gt // step] += 1
            pret_list[self.output_list[i] // step] += 1
        # except:
        #     print(gt // step)
        #     exit()
        avg_acc = np.array(tp_list) / total_list
        return avg_acc, pret_list, tp_list

    def confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix.

        Returns:
            A (n, n) matrix, where "n" is the number of classes.
        """
        self._merge()
        return metrics.confusion_matrix(self.merged_target, self.merged_output)
