#!/usr/bin/env python3

"""
@author: liying50
@since: 2023-01-05
"""

import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset

from torchstocks.common.trainer import BPTrainerWithDataset
from torchstocks.common.dataset import DataCollate
from torchstocks.utils.ema import ModelEMA
from torchstocks.utils.metrics import IouMeter

__all__ = [
    'SegmentationTrainer'
]


class SegmentationTrainer(BPTrainerWithDataset):
    """Segmentation trainer
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            train_collate: DataCollate,
            test_collate: DataCollate,
            optimizer: str = 'AdamW',
            batch_size: int = 16,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            eval_every_epoch: int = 5,
            shuffle=True,
            param_groups: list = None,
            pretrained: bool = True,
            clip_grad_norm: float = 0.1,
            lr_scheduler: str = 'LinearWarmupCosineDecay',
            lr_decay_min_value: float = 0.1,
            device: str = 'cpu',
            input_field: str = 'image',
            target_field: str = 'mask',
            output_dir: str = None
    ) -> None:
        super(SegmentationTrainer, self).__init__(
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
        self.eval_every_epoch = eval_every_epoch
        self.test_loader = self.auxiliary_loader
        self.pretrained = pretrained
        self.input_field = input_field
        self.target_field = target_field
        self.output_dir = output_dir
        self.ema = ModelEMA(self.model)

        self.status['progress'] = 0

    def train_step(self, image: torch.Tensor, targets: torch.Tensor):
        image = image.to(self.device)
        targets = targets.to(self.device)
        loss = self.model(inputs=image, targets=targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.ema.update(self.model)
        loss = loss.detach().cpu()
        return loss

    def predict_step(self, image: torch.Tensor):
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.ema.model(inputs=image, targets=None)
            return outputs

    def train(self):
        if self.train_loader is None:
            return

        self.status['loop'] = 0

        loss_g = None
        last_mIoU = 0
        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch + 1
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                image = doc[self.input_field]
                targets = doc[self.target_field]
                loss = self.train_step(image, targets)
                loss_g = 0.99 * loss_g + 0.01 * float(loss) if loss_g is not None else float(loss)
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
                print_string = ''
                print_string += f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f}'
                for k, v in self.status['metrics'].items():
                    print_string += f' {k}={v:.02%}'
                print(print_string)
                if self.output_dir is not None:
                    torch.save(self.ema.model, os.path.join(self.output_dir, 'last.pth'))
                    if 'mIoU' in self.status['metrics']:
                        if  self.status['metrics']['mIoU'] > last_mIoU:
                            torch.save(self.ema.model, os.path.join(self.output_dir, 'best.pth'))
                            last_mIoU = self.status['metrics']['mIoU']

    def evaluate(self):
        if self.test_loader is None:
            return

        meter = IouMeter(ignore_class=255, bg_class=-1)
        self.model.eval()
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            image, targets = doc[self.input_field], doc[self.target_field]
            outputs = self.predict_step(image)
            meter.update(output=outputs.cpu().numpy(), target=targets.cpu().numpy())
        miou_score, iou_dict = meter.m_iou()
        self.status['metrics'] = {'mIoU': miou_score}
        self.status['metrics'].update(iou_dict)
