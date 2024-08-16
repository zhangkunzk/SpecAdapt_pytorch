#!/usr/bin/env python3

"""
@author: xi
@since: 2022-02-25
"""
import numpy as np
from sklearn import metrics
from skimage import measure

__all__ = [
    'ADMeter'
]


class ADMeter(object):
    """Anomaly detetcion meter
    """

    def __init__(self):
        self.mask_list = []
        self.dist_map_list = []
        self.ab_score_list = []
        self.label_list = []

    def update(self, target: np.ndarray, output: np.ndarray) -> None:
        """
        Args:
            target: ground truth, should be the labeled image ndarray with shape of (n, h, w)
            output: model predict result image ndarray with shape of (n, h, w)

        """
        self.mask_list.extend(target)
        self.dist_map_list.extend(output)

    def update_img(self, label: np.ndarray, ab_score: np.ndarray) -> None:
        """

        Args:
            label: the ground truth label of image with shape (n, )
                    each label should be 0 or 1, 0 indicates normal sample and 1 otherwise.
            ab_score: model predict result the images' abnormality with shape (n, )
                    each score should be float.

        """
        self.label_list.extend(label)
        self.ab_score_list.extend(ab_score)

    def _merge(self) -> None:
        if len(self.dist_map_list) > 0:
            self.dist_map_list = np.array(self.dist_map_list)
            self.mask_list = np.array(self.mask_list)

    def auroc(self) -> float:
        """Compute pixel-level ROC score

        Returns: A float number represents pixel-level roc

        """
        self._merge()

        return metrics.roc_auc_score(
            self.mask_list.ravel(),
            self.dist_map_list.ravel()
        )

    def auroc_img(self):
        """Compute image-level ROC score

        Returns: A float number represents image-level roc

        """
        self.label_list = np.array(self.label_list)
        self.ab_score_list = np.array(self.ab_score_list)

        return metrics.roc_auc_score(
            self.label_list,
            self.ab_score_list
        )

    def compute_pro_curve(self, max_step=5000, expect_fpr=0.3) -> float:
        """Compute pro score.

        Args:
            max_step: The max step to threshold score
            expect_fpr: The fpr threshold
        Returns:
            A float number represents pro score

        The function may take a while, to reduce running time, set max step to lower value.
        """
        self._merge()

        pros_mean = []
        fpr_list = []
        thresholds = []

        max_val, min_val = self.dist_map_list.max(), self.dist_map_list.min()
        delta = (max_val - min_val) / max_step

        self.mask_list = np.zeros_like(self.mask_list, dtype=np.bool)
        self.mask_list[self.mask_list < 0.5] = 0
        self.mask_list[self.mask_list >= 0.5] = 1
        masks_neg = ~self.mask_list

        # varying thresholds
        for s in range(max_step):
            score_map = np.zeros_like(self.dist_map_list, dtype=bool)
            thr = max_val - s * delta
            thresholds.append(thr)

            score_map[self.dist_map_list <= thr] = 0
            score_map[self.dist_map_list > thr] = 1

            pro = []  # per-region overlap
            for i in range(1, len(self.dist_map_list)):
                true_label = measure.label(self.mask_list[i], connectivity=2)
                props = measure.regionprops(true_label)

                # for each defect
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                    cropped_pred_label = score_map[i][x_min:x_max, y_min:y_max]
                    cropped_mask = prop.filled_image
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)

            pros_mean.append(np.mean(pro))
            # fpr for pro-auc
            fpr = np.logical_and(masks_neg, score_map).sum() / masks_neg.sum()
            fpr_list.append(fpr)

        pros_mean = np.array(pros_mean)
        fpr_list = np.array(fpr_list)

        idx = fpr_list <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fpr_list[idx]
        fprs_selected = self.rescale(fprs_selected)  # rescale fpr [0, 0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]
        pro_auc_score = metrics.auc(fprs_selected, pros_mean_selected)
        return pro_auc_score

    @staticmethod
    def rescale(x):
        """Rescale
        """
        return (x - x.min()) / (x.max() - x.min())
