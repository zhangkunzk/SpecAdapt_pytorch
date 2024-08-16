#!/usr/bin/env python3

import math
import os
from typing import Union, Sequence, Tuple

import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from torchstocks.common.trainer import AbstractTrainer
from torchstocks.optim import CosineWarmupDecay
from torchstocks.utils.image import denormalize_image
from .model import Model


class SlidingWindow(object):

    def __init__(self, win_size: int, overlap_ratio: float = 0.05):
        self.win_size = win_size
        self.overlap = int(win_size * overlap_ratio)

    def __call__(self, size: int) -> Sequence[Tuple[int, int, int]]:
        if size <= self.win_size:
            return [(0, size, 0)]

        win_size = self.win_size
        overlap = self.overlap
        num_wins = math.floor((size - win_size) / (win_size - overlap) + 0.5) + 1
        total = win_size + (num_wins - 1) * (win_size - overlap)
        remains = size - total
        # print(win_size, num_wins, total, remains)
        if remains == 0:
            num_big_overlaps = total - size
            num_small_overlaps = num_wins - 1
        elif remains > 0:
            # print('window expand')
            win_size += math.ceil(remains / num_wins)
            total = win_size + (num_wins - 1) * (win_size - overlap)
            num_big_overlaps = total - size
            num_small_overlaps = num_wins - 1 - num_big_overlaps
        else:
            # print('window squeeze')
            win_size -= math.floor(-remains / num_wins)
            total = win_size + (num_wins - 1) * (win_size - overlap)
            num_big_overlaps = total - size
            num_small_overlaps = num_wins - 1 - num_big_overlaps

        overlaps = [self.overlap + 1] * num_big_overlaps + [self.overlap] * num_small_overlaps
        win_list = [(0, win_size, 0)]
        pos = win_size
        for i in range(num_wins - 1):
            pos -= overlaps[i]
            win_list.append((pos, pos + win_size, overlaps[i]))
            pos += win_size
        return win_list


class ADModelWrapper(object):

    def __init__(self, model, max_size=1200, win_size=512, overlap_ratio=0.05):
        self.model = model
        self.sliding_win = SlidingWindow(win_size, overlap_ratio)
        self.max_area = max_size + max_size

    def __call__(self, x, task=0):
        return self.forward(x, task)

    def forward(self, x, task=0):
        n, _, h, w = x.shape
        if h * w <= self.max_area:
            return self.model.forward(x, task)

        dtype, device = x.dtype, x.device
        h_win_list = self.sliding_win(h)
        w_win_list = self.sliding_win(w)
        heatmap = torch.zeros((n, 1, h, w), dtype=dtype, device=device)
        ab_score = torch.zeros((n,), dtype=dtype, device=device)
        for row in h_win_list:
            y1, y2, yo = row
            ya = torch.linspace(0, 1, yo, dtype=dtype, device=device).reshape((1, 1, -1, 1)) if yo > 0 else None
            for col in w_win_list:
                x1, x2, xo = col
                xa = torch.linspace(0, 1, xo, dtype=dtype, device=device).reshape((1, 1, 1, -1)) if xo > 0 else None
                _x = x[:, :, y1:y2, x1:x2]
                _heatmap, _ab_score = self.model.forward(_x, task)
                heatmap[:, :, y1 + yo:y2, x1 + xo:x2] = _heatmap[:, :, yo:, xo:]
                # print(_x.shape, _heatmap.shape)
                if ya is not None:
                    heatmap[:, :, y1:y1 + yo, x1:x2].mul_(1 - ya).add_(_heatmap[:, :, :yo, :].mul_(ya))
                if xa is not None:
                    heatmap[:, :, y1:y2, x1:x1 + xo].mul_(1 - xa).add_(_heatmap[:, :, :, :xo].mul_(xa))
                ab_score[:] = torch.maximum(ab_score, _ab_score)
        return heatmap, ab_score

    def update_memory(self, x, task=0, lr=None):
        n, _, h, w = x.shape
        if h * w <= self.max_area:
            return self.model.update_memory(x, task, lr)

        h_win_list = self.sliding_win(h)
        w_win_list = self.sliding_win(w)
        ret = None
        for row in h_win_list:
            for col in w_win_list:
                y1, y2, yo = row
                x1, x2, xo = col
                _x = x[:, :, y1:y2, x1:x2]
                ret = self.model.update_memory(_x, task, lr)
        return ret


class Trainer(AbstractTrainer):

    def __init__(
            self,
            model: Model = None,
            train_dataset=None,
            test_dataset=None,
            batch_size: int = 8,
            max_lr: float = 0.8,
            num_epochs: int = 10,
            num_workers: int = 10,
            output_dir=None,
            device=None,
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.device = device

        self.model.to(self.device)
        self.wrapper = ADModelWrapper(self.model)

        if len(self.train_dataset) < 1600:
            n = math.ceil(1600.0 / len(self.train_dataset))
            self.train_dataset = ConcatDataset([self.train_dataset] * n)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        ) if self.train_dataset else None
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        ) if self.test_dataset else None

        self.num_loops = self.num_epochs * len(self.train_loader)
        self.lr_decay = CosineWarmupDecay(self.num_loops)

    def train_step(self, x: torch.Tensor, lr: float):
        x = x.to(self.device)  # (n, c, h, w)
        self.wrapper.update_memory(x, lr=lr)

    def predict_step(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)  # (n, c, h, w)
            heatmap, ab_score = self.wrapper(x)
            ab_score = ab_score.detach().cpu()
            heatmap = heatmap.detach().cpu()
        return heatmap, ab_score

    def train(self):
        self.status['training'] = True
        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch
            self.model.train()
            self.model.backbone.eval()
            train_loop = tqdm(self.train_loader, leave=False, ncols=96)
            for i, doc in enumerate(train_loop):
                loop = epoch * len(self.train_loader) + i
                self.status['loop'] = loop
                image = doc['image']
                lr = self.max_lr * self.lr_decay[loop]
                self.train_step(image, lr)
                train_loop.set_description(f'[{epoch + 1}/{self.num_epochs}] LR={lr:.05f}', False)

            metrics = self.evaluate()
            print_string = ''
            print_string += f'[{epoch + 1}/{self.num_epochs}]'
            if metrics is not None:
                if metrics['pixel_au_roc'] is not None:
                    print_string += f' pixel_au_roc={metrics["pixel_au_roc"]:.06f}'
                if metrics['au_roc'] is not None:
                    print_string += f' au_roc={metrics["au_roc"]:.06f}'
                if metrics['threshold'] is not None:
                    p, r, t = metrics['threshold']
                    print_string += f' precision={p:.02%} recall={r:.02%} threshold={t:0.6f}'
            print(print_string)

            if self.output_dir is not None:
                torch.save(self.model, os.path.join(self.output_dir, f'epoch_{epoch + 1}.pth'))
        self.status['training'] = False

    def evaluate(self, target_recall=0.95, target_precision=None):
        if self.test_loader is None:
            return

        self.status['evaluating'] = True
        meter = AnomalyDetectionMeter()
        self.model.eval()
        test_loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in test_loop:
            image = doc['image']
            label = doc['label']
            heatmap, ab_score = self.predict_step(image)

            meter.update(
                heatmap=heatmap.numpy(),
                ab_score=ab_score.numpy(),
                label=label.numpy(),
                mask=doc['mask'].numpy() if 'mask' in doc else None,
                image=[denormalize_image(_i, transpose=True) for _i in image.numpy()]
            )

        metrics = {
            'au_roc': meter.au_roc(),
            'pixel_au_roc': meter.pixel_au_roc(),
            'pr_curve': meter.precision_recall_curve()
        }
        if target_recall is not None:
            threshold = meter.find_threshold_by_recall(target_recall)
        elif target_precision is not None:
            threshold = meter.find_threshold_by_precision(target_precision)
        else:
            threshold = meter.find_threshold_by_f1()
        metrics['threshold'] = threshold

        if self.output_dir is not None:
            epoch = self.status['epoch'] if ('epoch' in self.status) else 0
            meter.write_images(os.path.join(self.output_dir, f'epoch_{epoch + 1}'), threshold[2])

        self.status['metrics'] = metrics
        self.status['evaluating'] = False
        return metrics


class AnomalyDetectionMeter(object):

    def __init__(self):
        self.heatmap_list = []
        self.ab_score_list = []
        self.label_list = []
        self.mask_list = []
        self.image_list = []

        self._au_roc = None
        self._pixel_au_roc = None
        self._precision_recall_curve = None

    def update(
            self,
            heatmap: Union[np.ndarray, Sequence[np.ndarray]],  # (n, 1, h, w)
            ab_score: Union[np.ndarray, Sequence[float]],  # (n,)
            label: Union[np.ndarray, Sequence[int]],  # (n,)
            mask: Union[np.ndarray, Sequence[np.ndarray], None] = None,  # (n, 1, h, w)
            image: Union[np.ndarray, Sequence[np.ndarray], None] = None  # (n, c, h, w)
    ) -> None:
        self._au_roc = None
        self._pixel_au_roc = None
        self._precision_recall_curve = None

        self.heatmap_list.extend(heatmap)
        self.ab_score_list.extend(float(_s) for _s in ab_score)
        self.label_list.extend((0 if int(_l) == 0 else 1) for _l in label)
        if mask is not None:
            self.mask_list.extend(mask)
        if image is not None:
            self.image_list.extend(image)

    def au_roc(self) -> float:
        if self._au_roc is None:
            self._au_roc = roc_auc_score(
                y_true=np.array(self.label_list, np.int64).ravel(),
                y_score=np.array(self.ab_score_list, np.float32).ravel()
            )
        return self._au_roc

    def pixel_au_roc(self) -> float:
        if self._pixel_au_roc is None:
            self._pixel_au_roc = roc_auc_score(
                y_true=np.array(self.mask_list, np.float32).ravel(),
                y_score=np.array(self.heatmap_list, np.float32).ravel()
            ) if len(self.mask_list) == len(self.heatmap_list) else None
        return self._pixel_au_roc

    def precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._precision_recall_curve is None:
            precision, recall, threshold = precision_recall_curve(
                np.array(self.label_list, np.int64).ravel(),
                np.array(self.ab_score_list, np.float32).ravel()
            )
            threshold = np.concatenate([threshold, threshold[-1:]], 0)
            self._precision_recall_curve = (precision, recall, threshold)
        return self._precision_recall_curve

    def find_threshold_by_recall(self, target_recall: float = 0.95) -> Tuple[float, float, float]:
        precision, recall, threshold = self.precision_recall_curve()

        target_recall = min(target_recall, np.max(recall))
        allow = recall >= target_recall

        allowed_recall = recall[allow]
        allowed_precision = precision[allow]
        allowed_threshold = threshold[allow]

        idx = np.argmax(allowed_precision)

        best_precision = allowed_precision[idx]
        best_recall = allowed_recall[idx]
        best_threshold = allowed_threshold[idx]
        return best_precision, best_recall, best_threshold

    def find_threshold_by_precision(self, target_precision: float = 0.95) -> Tuple[float, float, float]:
        precision, recall, threshold = self.precision_recall_curve()

        target_precision = min(target_precision, np.max(recall))
        allow = precision >= target_precision

        allowed_recall = recall[allow]
        allowed_precision = precision[allow]
        allowed_threshold = threshold[allow]

        idx = np.argmax(allowed_recall)

        best_precision = allowed_precision[idx]
        best_recall = allowed_recall[idx]
        best_threshold = allowed_threshold[idx]
        return best_precision, best_recall, best_threshold

    def find_threshold_by_f1(self) -> Tuple[float, float, float]:
        precision, recall, threshold = self.precision_recall_curve()
        f1 = 2 * (precision * recall) / (precision + recall)
        idx = np.argmax(f1)
        return precision[idx], recall[idx], threshold[idx]

    def write_images(self, output_dir: str, threshold: float):
        if len(self.image_list) != len(self.ab_score_list):
            return

        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(self.image_list)):
            image = self.image_list[i]
            heatmap = self.heatmap_list[i]
            ab_score = self.ab_score_list[i]
            label = self.label_list[i]

            target = 'OK' if label == 0 else 'NG'
            if ab_score >= threshold:
                predict = 'NG'
                heatmap = heatmap / np.max(heatmap)
                heatmap = np.clip(heatmap * 255, 0, 255)
                heatmap = np.array(heatmap, dtype=np.uint8)
            else:
                predict = 'OK'
                heatmap = np.zeros_like(heatmap, np.uint8)
            heatmap = np.transpose(heatmap, (1, 2, 0))
            heatmap = np.tile(heatmap, (1, 1, 3))
            image_combine = [image, heatmap]
            image_combine = np.concatenate(image_combine, 1)
            image_combine = np.flip(image_combine, 2)  # RGB to BGR
            output_file = os.path.join(output_dir, f'{target}-{predict}-{ab_score:.04f}_{i}.jpg')
            cv.imwrite(output_file, image_combine)


def find_threshold_by_recall(label, score, target_recall=0.95, step=0.1, eps=1e-3):
    """This function is written for lizard.
    todo: It should be merged into the AnomalyDetectionMeter in the future.
    """
    precision_list, recall_list, threshold_list = [], [], []
    start = float(np.min(score))
    end = float(np.max(score))
    thr = start
    while thr <= end:
        output = np.array(score > thr, np.int64)
        precision = precision_score(label, output)
        recall = recall_score(label, output)
        precision_list.append(precision)
        recall_list.append(recall)
        threshold_list.append(thr)
        thr += step

    precision = np.array(precision_list)
    recall = np.array(recall_list)
    threshold = np.array(threshold_list)

    allow = recall > target_recall
    allowed_precision = precision[allow]
    allowed_threshold = threshold[allow]
    max_allowed_precision = np.max(allowed_precision)
    candidate = (allowed_precision <= max_allowed_precision) & (allowed_precision > max_allowed_precision - eps)
    final_threshold = allowed_threshold[candidate].mean()
    return final_threshold
