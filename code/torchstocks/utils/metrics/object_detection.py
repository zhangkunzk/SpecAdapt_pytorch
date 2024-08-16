#!/usr/bin/env python3

"""
@author: xi
@since: 2021-12-16
"""

import collections
from typing import Union, Tuple, Dict

import multiprocessing as mp
import numpy as np

__all__ = [
    'APMeter',
    'APMeterV2',
    'MAPMeter'
]


class APMeter(object):
    """Average precision meter
    """

    def __init__(self, iou_threshold: float, data_format):
        self.iou_threshold = iou_threshold
        self.data_format = data_format
        self.target_dict = collections.defaultdict(list)
        self.output_list = []

        self._computed = False
        self._score = None
        self._matching = None
        self._precision = None
        self._recall = None
        self._f1 = None
        self._f1_curve = None
        self._ap = None

    def add_target(self, name: Union[str, int], bboxes: np.ndarray):
        """Add a target bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 5) in xywhc format.
        """
        self._computed = False
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 5
            for bbox in bboxes:
                self.target_dict[name].append(bbox)
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 5
            self.target_dict[name].append(bboxes)
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def add_output(self, name: Union[str, int], bboxes: np.ndarray):
        """Add an output bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 6) in xywhcs format.
        """
        self._computed = False
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 6
            for bbox in bboxes:
                score = float(bbox[5])
                self.output_list.append([name, bbox, score])
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 6
            score = float(bboxes[5])
            self.output_list.append([name, bboxes, score])
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def compute(self):
        """Compute
        """
        if self._computed:
            return (
                self._score,
                self._matching,
                self._precision,
                self._recall,
                self._f1,
                self._f1_curve,
                self._ap
            )

        if len(self.output_list) == 0 or len(self.target_dict) == 0:
            # self._computed = True
            self._score = np.array([], dtype=np.float32)
            self._matching = np.array([], dtype=np.float32)
            self._precision = np.array([], dtype=np.float32)
            self._recall = np.array([], dtype=np.float32)
            self._f1 = np.array([], dtype=np.float32)
            self._f1_curve = np.zeros(1000, dtype=np.float32)
            self._ap = 0.0
            # return
        else:
            self._score, self._matching = self._compute_matching_vector()
            self._precision, self._recall = self._compute_pr_curve(self._matching)
            self._f1 = 2 * self._precision * self._recall / (self._precision + self._recall + 1e-16)
            ap = 0.0
            last_r = 0.0
            for i, _ in enumerate(self._recall):
                r_i, p_i = self._recall[i], self._precision[i]
                ap += (r_i - last_r) * p_i
                last_r = r_i
            self._ap = ap
            x_axis = np.linspace(0, 1, 1000)
            conf = self._score[::-1]
            f1 = self._f1[::-1]
            self._f1_curve = np.interp(x_axis, conf.astype(np.float32), f1.astype(np.float32)).astype(np.float32)
        self._computed = True
        return (
            self._score,
            self._matching,
            self._precision,
            self._recall,
            self._f1,
            self._f1_curve,
            self._ap
        )

    def _compute_matching_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        self.output_list.sort(key=lambda _t: -_t[2])
        score = np.array([_t[2] for _t in self.output_list], dtype=np.ndarray)

        matching = np.zeros((len(self.output_list),), dtype=np.float32)
        matched_targets = set()
        for i, (name, bbox_output, _) in enumerate(self.output_list):
            for bbox_target in self.target_dict[name]:
                if id(bbox_target) in matched_targets:
                    continue
                if self._iou(bbox_output, bbox_target, self.data_format) >= self.iou_threshold:
                    matching[i] = 1
                    matched_targets.add(id(bbox_target))
                    break
        return score, matching

    def _compute_pr_curve(self, matching_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tp = np.cumsum(matching_vec)  # num of the correctly predicted positives
        tp_fp = np.arange(1, len(matching_vec) + 1, dtype=np.float32)  # num of the predicted positives
        tp_fn = sum([len(_v) for _v in self.target_dict.values()])  # num of the true positives
        precision = tp / tp_fp
        recall = tp / tp_fn

        # smooth the pr curve
        for i in range(len(precision) - 1):
            precision[i] = precision[i + 1:].max()
        return precision, recall

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray, data_format, eps=1e-7) -> float:
        if data_format == 'xywh':
            b1_x, b1_y, b1_w, b1_h = bbox1[0], bbox1[1], bbox1[2] * 0.5, bbox1[3] * 0.5
            b2_x, b2_y, b2_w, b2_h = bbox2[0], bbox2[1], bbox2[2] * 0.5, bbox2[3] * 0.5
            b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
            b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h
        elif data_format == 'xyxy':
            b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        else:
            raise RuntimeError(f'Unsupported data format "{data_format}".')
        w_inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0)
        h_inter = max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)
        area_inter = w_inter * h_inter

        area_a = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        area_union = area_a + area_b - area_inter

        iou = area_inter / (area_union + eps)
        return iou

    def score(self) -> np.ndarray:
        """Compute score
        """
        self.compute()
        return self._score

    def matching(self) -> np.ndarray:
        """Compute matching
        """
        self.compute()
        return self._matching

    def precision(self) -> np.ndarray:
        """Compute precision
        """
        self.compute()
        return self._precision

    def recall(self) -> np.ndarray:
        """Compute recall
        """
        self.compute()
        return self._recall

    def ap(self) -> float:
        """Compute Average Precision (AP) score.

        Returns:
            A float number represents the AP score.
        """
        self.compute()
        return self._ap

    def f1(self) -> np.ndarray:
        """Compute f1-score
        """
        self.compute()
        return self._f1

    def f1_curve(self) -> np.ndarray:
        """Compute f1-score curve
        """
        self.compute()
        return self._f1_curve


class APMeterV2(object):
    """Average Precision meter version2
    """

    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold
        self.target_dict = collections.defaultdict(list)
        self.output_dict = collections.defaultdict(list)

    def add_target(self, name: Union[str, int], bboxes: np.ndarray):
        """Add a target bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 5) in xywhc format.
        """
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 5
            for bbox in bboxes:
                self.target_dict[name].append(bbox)
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 5
            bbox = bboxes
            self.target_dict[name].append(bbox)
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def add_output(self, name: Union[str, int], bboxes: np.ndarray):
        """Add an output bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 6) in xywhcs format.
        """
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 6
            for bbox in bboxes:
                score = float(bbox[5])
                self.output_dict[name].append([bbox, score])
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 6
            bbox = bboxes
            score = float(bbox[5])
            self.output_dict[name].append((bbox, score))
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def ap(self) -> float:
        """Compute Average Precision (AP) score.

        Returns:
            A float number represents the AP score.
        """
        if len(self.output_dict) == 0 or len(self.target_dict) == 0:
            return 0.0

        matching_vec = self._compute_matching_vector()
        precision, recall = self._compute_pr_curve(matching_vec)
        ap = 0.0
        last_r = 0.0
        for i, _ in enumerate(recall):
            r_i, p_i = recall[i], precision[i]
            ap += (r_i - last_r) * p_i
            last_r = r_i
        return ap

    def _compute_matching_vector(self) -> np.ndarray:
        output_list = []
        for name in self.output_dict:
            # bboxes_output: (m, 4)
            outputs = self.output_dict[name]
            bboxes_output = np.zeros((len(outputs), 4), dtype=np.float32)
            scores = np.zeros((len(outputs),), dtype=np.float32)
            for i, (bbox, score) in enumerate(outputs):
                bboxes_output[i] = bbox[:4]
                scores[i] = score

            # bboxes_target: (n, 4)
            targets = self.target_dict[name]
            bboxes_target = np.zeros((len(targets), 4), dtype=np.float32)
            for i, bbox in enumerate(targets):
                bboxes_target[i] = bbox[:4]

            # iou: (m, n)
            print('******************************')
            iou = APMeterV2._iou(bboxes_output, bboxes_target)
            indices_output, indices_target = np.where(iou >= self.iou_threshold)
            matches = [[] for _ in range(len(outputs))]
            for i, j in zip(indices_output, indices_target):
                matches[i].append(j)
            for match, score in zip(matches, scores):
                output_list.append((name, match, score))
        output_list.sort(key=lambda _t: -_t[2])

        matching_vec = np.zeros((len(output_list),), dtype=np.float32)
        matched = collections.defaultdict(set)
        for i, (name, match, _) in enumerate(output_list):
            for j in match:
                if j not in matched[name]:
                    matched[name].add(j)
                    matching_vec[i] = 1
                    break
        return matching_vec

    def _compute_pr_curve(self, matching_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tp = np.cumsum(matching_vec)  # num of the correctly predicted positives
        tp_fp = np.arange(1, len(matching_vec) + 1, dtype=np.float32)  # num of the predicted positives
        tp_fn = sum([len(_v) for _v in self.target_dict.values()])  # num of the true positives
        precision = tp / tp_fp
        recall = tp / tp_fn

        # smooth the pr curve
        for i in range(len(precision) - 1):
            precision[i] = precision[i + 1:].max()
        return precision, recall

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray, eps=1e-7) -> float:
        b1_x, b1_y, b1_w, b1_h = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2] * 0.5, bbox1[:, 3] * 0.5
        b2_x, b2_y, b2_w, b2_h = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2] * 0.5, bbox2[:, 3] * 0.5
        b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
        b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h

        w_inter = np.minimum(b1_x2[:, None], b2_x2[None, :]) - np.maximum(b1_x1[:, None], b2_x1[None, :])
        w_inter = np.maximum(w_inter, 0)
        h_inter = np.minimum(b1_y2[:, None], b2_y2[None, :]) - np.maximum(b1_y1[:, None], b2_y1[None, :])
        h_inter = np.maximum(h_inter, 0)
        area_inter = w_inter * h_inter

        area_1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        ares_2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        area_union = area_1[:, None] + ares_2[None, :] - area_inter

        iou = area_inter / (area_union + eps)
        return iou


class MAPMeter(object):
    """Mean average precision meter
    """

    def __init__(self, iou_threshold: float = 0.5, num_workers: int = 0.5, data_format='xywh'):
        """Mean Average Precision (MAP) Meter.

        Args:
            iou_threshold: A threshold used to determined whether the prediction bbox is correct.
        """
        self.ap_meters = collections.defaultdict(lambda: APMeter(iou_threshold, data_format))
        if num_workers is None:
            self.num_workers = 1
        elif isinstance(num_workers, float):
            assert num_workers <= 1
            self.num_workers = max(int(num_workers * mp.cpu_count()), 4)
        else:
            self.num_workers = num_workers

    def update(
            self,
            name: Union[str, int],
            output: np.ndarray,
            target: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add output (model prediction) and target (ground truth) for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            output: (n, 6) matrix in xywhcs format.
            target: (n, 5) matrix in xywhc format.
            label: The labels of the bboxes.
        """
        self.update_output(name, output, label)
        self.update_target(name, target, label)

    def update_output(
            self,
            name: Union[str, int],
            output: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add output bounding boxes for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            output: (n, 6) matrix in xywhcs format.
            label: The labels of the bboxes.
        """
        assert len(output.shape) == 2 and output.shape[1] >= 6
        if label is None:
            for bbox in output:
                self.ap_meters[int(bbox[4])].add_output(name, bbox)
        elif isinstance(label, np.ndarray):
            assert label.shape[0] == output.shape[0]
            for bbox, l in zip(output, label):
                self.ap_meters[int(l)].add_output(name, bbox)
        elif isinstance(label, int):
            for bbox in output:
                self.ap_meters[label].add_output(name, bbox)
        else:
            raise RuntimeError(f'Invalid label type {type(label)}')

    def update_target(
            self,
            name: Union[str, int],
            target: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add target bounding boxes for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            target: (n, 5) matrix in xywhc format.
            label: The labels of the bboxes.
        """
        assert len(target.shape) == 2 and target.shape[1] >= 5
        if label is None:
            for bbox in target:
                self.ap_meters[int(bbox[4])].add_target(name, bbox)
        elif isinstance(label, np.ndarray):
            assert label.shape[0] == target.shape[0]
            for bbox, l in zip(target, label):
                self.ap_meters[int(l)].add_target(name, bbox)
        elif isinstance(label, int):
            for bbox in target:
                self.ap_meters[label].add_target(name, bbox)
        else:
            raise RuntimeError(f'Invalid label type {type(label)}')

    def m_ap(self) -> Tuple[float, Dict[int, float]]:
        """Compute MAP score for all classes.

        Returns:
            A float number that represents the MAP score and a dict for ap scores in each class.
        """
        if len(self.ap_meters) == 0:
            return 0.0, {}
        ap_dict = self.ap()
        score = float(np.mean(list(ap_dict.values())))
        return score, ap_dict

    def m_f1_curve(self) -> Dict[int, np.ndarray]:
        """Compute f1-score in different confidence
        """
        if len(self.ap_meters) == 0:
            return {}
        f1_curve_dict = self.f1_curve()
        mean_f1 = np.mean(np.array(list(f1_curve_dict.values())), axis=0)
        f1_curve_dict.update({'mean': mean_f1})
        return f1_curve_dict

    def score(self) -> Dict[int, np.ndarray]:
        """Compute score
        """
        self._compute()
        return {c: m.score() for c, m in self.ap_meters.items()}

    def matching(self) -> Dict[int, np.ndarray]:
        """Compute matching
        """
        self._compute()
        return {c: m.matching() for c, m in self.ap_meters.items()}

    def precision(self) -> Dict[int, np.ndarray]:
        """Compute precision
        """
        self._compute()
        return {c: m.precision() for c, m in self.ap_meters.items()}

    def recall(self) -> Dict[int, np.ndarray]:
        """Compute recall
        """
        self._compute()
        return {c: m.recall() for c, m in self.ap_meters.items()}

    def ap(self) -> Dict[int, np.ndarray]:
        """Compute average precision
        """
        self._compute()
        return {c: m.ap() for c, m in self.ap_meters.items()}

    def f1(self) -> Dict[int, np.ndarray]:
        """Compute f1-score
        """
        self._compute()
        return {c: m.f1() for c, m in self.ap_meters.items()}

    def f1_curve(self) -> Dict[int, np.ndarray]:
        """Compute f1-score curve
        """
        self._compute()
        return {c: m.f1_curve() for c, m in self.ap_meters.items()}

    def _compute(self):
        if self.num_workers <= 1:
            for meter in self.ap_meters.values():
                meter.compute()
        else:
            meter_list = self.ap_meters.values()

            with mp.Pool(self.num_workers) as pool:
                computed_list = [pool.apply_async(meter.compute) for meter in meter_list]
                computed_list = [computed.get() for computed in computed_list]

            for meter, computed in zip(meter_list, computed_list):
                (meter._score,
                 meter._matching,
                 meter._precision,
                 meter._recall,
                 meter._f1,
                 meter._f1_curve,
                 meter._ap) = computed
                meter._computed = True
