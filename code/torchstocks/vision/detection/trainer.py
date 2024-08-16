#!/usr/bin/env python3

"""
@author: liying50
@since: 2022-11-03
"""

from typing import List, Tuple
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.common.dataset import DataCollate
from torchstocks.common.trainer import BPTrainerWithDataset
from torchstocks.utils.ema import ModelEMA
from torchstocks.utils.metrics import MAPMeter
from torchstocks.vision.detection.model.yolov5 import YoloDecoder
from torchstocks.vision.detection.model.rcnn.decoder import FastRCNNDecoder


__all__ = [
    'DetectionTrainer'
]


class DetectionTrainer(BPTrainerWithDataset):
    """Detection trainer
    """
    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            train_collate: DataCollate,
            test_collate: DataCollate,
            optimizer: str = 'AdamW',
            batch_size: int = 32,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 300,
            num_workers: int = 10,
            eval_every_epoch: int = 5,
            shuffle=True,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            lr_scheduler: str = 'LinearWarmupCosineDecay',
            lr_decay_min_value: float = 0.1,
            device: str = 'cpu',
            input_field: str = 'image',
            target_field: str = 'bboxes',
            output_dir: str = None,
            train_obj_threshold: float = 0.001,
            train_nms_threshold: float = 0.6
    ) -> None:
        super(DetectionTrainer, self).__init__(
            model=model,
            train_dataset=train_dataset,
            auxiliary_dataset=test_dataset,
            train_collate=train_collate,
            auxiliary_collate=test_collate,
            optimizer=optimizer,
            batch_size=batch_size,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            shuffle=shuffle,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            lr_scheduler=lr_scheduler,
            lr_decay_min_value=lr_decay_min_value,
            device=device
        )
        if hasattr(self.model, 'num_heads'):  # yolo
            print('yolo')
            self.yolo = True
            self.rcnn = False
            decoder = YoloDecoder(
                obj_threshold=train_obj_threshold,
                nms_threshold=train_nms_threshold
            )
        elif hasattr(self.model, 'in_features'):  # faster_rcnn
            print('faster_rcnn')
            self.yolo = False
            self.rcnn = True
            decoder = FastRCNNDecoder(
                score_threshold=train_obj_threshold,
                nms_threshold=train_nms_threshold,
                topk_per_image=100
            )
        else:
            print('error')
        self.decoder = decoder.to(self.device)
        assert self.yolo != self.rcnn
        self.eval_every_epoch = eval_every_epoch
        self.test_loader = self.auxiliary_loader
        self.input_field = input_field
        self.target_field = target_field
        self.output_dir = output_dir
        self._init_model()
        self.ema = ModelEMA(self.model)
        self.status['progress'] = 0

    def train_step(self, image: torch.Tensor, bboxes: List) -> Tuple:
        """Train
        """
        image = image.to(self.device)
        for i, bbox in enumerate(bboxes):
            bboxes[i] = bbox.to(self.device)

        if self.yolo:
            loss, loss_obj, loss_box, loss_cls = self.model(inputs=image, targets=bboxes)
        else:
            loss_dict = self.model(inputs=image, targets=bboxes)
            loss = sum(loss_dict.values())
            assert torch.isfinite(loss).all(), loss_dict
            _loss_dict = {k: f'{v.item():.06f}' for k, v in loss_dict.items()}
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.ema.update(self.model)
        if self.yolo:
            loss = loss.detach().cpu()
            loss_obj = loss_obj.detach().cpu()
            loss_box = loss_box.detach().cpu()
            loss_cls = loss_cls.detach().cpu()
            return loss, loss_obj, loss_box, loss_cls
        else:
            loss_cls = _loss_dict['loss_cls']
            loss_box_reg = _loss_dict['loss_box_reg']
            loss_rpn_cls = _loss_dict['loss_rpn_cls']
            loss_rpn_loc = _loss_dict['loss_rpn_loc']
            return loss_cls, loss_box_reg, loss_rpn_cls, loss_rpn_loc

    def predict_step(self, image: torch.Tensor) -> List:
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.ema.model(inputs=image, targets=None)
            if self.yolo:
                outputs = self.decoder(outputs)
            else:
                assert self.rcnn
                boxes, scores, image_shapes = outputs
                outputs = self.decoder(boxes, scores, image_shapes)
                outputs = [torch.cat((res['pred_boxes'], res['pred_classes'][:, None],
                                    res['scores'][:, None]), dim=1) for res in outputs]
            outputs = [output.detach().cpu() for output in outputs]
            return outputs

    def train(self):
        if self.train_loader is None:
            return

        self.status['loop'] = 0

        loss_g, ag, bg, cg = None, None, None, None
        last_mAP = 0
        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch + 1
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                image = doc[self.input_field]
                bboxes = doc[self.target_field]
                loss, a, b, c = self.train_step(image, bboxes)

                loss_g = 0.99 * loss_g + 0.01 * float(loss) if loss_g is not None else float(loss)
                ag = 0.99 * ag + 0.01 * float(a) if ag is not None else float(a)
                bg = 0.99 * bg + 0.01 * float(b) if bg is not None else float(b)
                cg = 0.99 * cg + 0.01 * float(c) if cg is not None else float(c)
                lr = self.optimizer.param_groups[0]['lr']

                self.status['loop'] += 1
                self.status['loss'] = loss
                self.status['loss_g'] = loss_g
                self.status['lr'] = lr

                info = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)

            self.status['progress'] = self.status['epoch'] / self.num_epochs
            if (epoch + 1) % self.eval_every_epoch != 0 and (epoch + 1) != self.num_epochs:
                continue

            self.evaluate()
            if 'metrics' in self.status:
                if 'AP50' in self.status['metrics']:
                    print_string = ''
                    print_string += f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f}'
                    for k, v in self.status['metrics']['AP50'].items():
                        print_string += f' {k}={v:.02%}'
                    print(print_string)
                if self.output_dir is not None:
                    if hasattr(self.ema.model, 'sync_grads'):
                        self.ema.model.sync_grads = None
                    torch.save(self.ema.model, os.path.join(self.output_dir, 'last.pth'))
                    if 'mAP' in self.status['metrics']['AP50']:
                        if self.status['metrics']['AP50']['mAP'] > last_mAP:
                            torch.save(self.ema.model, os.path.join(self.output_dir, 'best.pth'))
                            last_mAP = self.status['metrics']['AP50']['mAP']

    def evaluate(self):
        if self.test_loader is None:
            return

        if self.yolo:
            data_format = 'xywh'
        else:
            assert self.rcnn
            data_format='xyxy'
        meter = MAPMeter(0.5, data_format=data_format)
        self.model.eval()
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            filename = doc['filename']
            image, bboxes = doc[self.input_field], doc[self.target_field]
            outputs = self.predict_step(image)
            bboxes = [bbox.numpy() for bbox in bboxes]
            outputs = [output.numpy() for output in outputs]
            for idx, file_name in enumerate(filename):
                meter.update(file_name, output=outputs[idx], target=bboxes[idx])
        map_score, ap_dict = meter.m_ap()
        f1_curve_dict = meter.m_f1_curve()
        ap_dict.update({'mAP': map_score})
        self.status['metrics'] = {'AP50': ap_dict}
        self.status['metrics'].update({'f1_score': f1_curve_dict})
        # self.status['metrics'] 数据格式：
            # {
            #     'AP50': {0: float, 1: float, ..., 'mAP': float},
            #     'f1_score': {0: np.ndarray, 1: np.ndarray, ..., 'mean': np.ndarray}
            # }
