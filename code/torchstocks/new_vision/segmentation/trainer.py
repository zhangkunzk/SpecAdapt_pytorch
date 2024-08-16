#!/usr/bin/env python3

"""
@author: liying50
@since: 2023-01-05
"""

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset

from torchstocks.new_common.trainer import AbstractTrainer
from torchstocks.common.dataset import DataCollate

__all__ = [
    'SegmentationTrainer'
]


class SegmentationTrainer(AbstractTrainer):
    """Segmentation trainer
    """

    def __init__(
            self,
            model: nn.Module,
            ema: nn.Module,
            train_dataset: Dataset,
            train_collate: DataCollate,
            batch_size: int = 16,
            num_workers: int = 10,
            shuffle: bool = True,
            input_field: str = 'image',
            target_field: str = 'mask',
            optimizer: str = 'AdamW',
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            lr_scheduler: str = 'LinearWarmupCosineDecay',
            lr_decay_min_value: float = 0.1,
            device: str = 'cpu',
            epoch_callback=None
    ) -> None:
        super(SegmentationTrainer, self).__init__()

        # model options
        self.model = self.init_model(model, param_groups, device)
        self.ema = ema

        # dataset options
        self.input_field = input_field
        self.target_field = target_field
        self.train_dataset, self.train_loader, _ = self.init_train_dataset(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=train_collate
        )
        assert batch_size <= len(self.train_dataset)

        # optimizer options
        self.num_epochs = num_epochs
        if self.train_loader:
            self.num_loops = len(self.train_loader) * self.num_epochs
            self.optimizer = self.init_optimizer(
                optimizer=optimizer,
                model=self.model,
                param_groups=param_groups,
                lr=max_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                clip_grad_norm=clip_grad_norm
            )
            self.lr_scheduler = self.init_lr_scheduler(
                optimizer=self.optimizer,
                scheduler_type=lr_scheduler,
                num_loops=self.num_loops,
                min_value=lr_decay_min_value
            )

        # misc
        self.device = device
        self.epoch_callback = epoch_callback
        self.status['progress'] = 0

    def _train(self, image: torch.Tensor, targets: torch.Tensor):
        image = image.to(self.device)
        targets = targets.to(self.device)
        loss = self.model(inputs=image, targets=targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.ema.update(self.model)
        loss = loss.detach().cpu()
        return loss

    def run(self):
        """Run
        """
        if self.train_loader is None:
            return

        self.status['loop'] = 0
        self.status['best_mIoU'] = 0
        self.status['num_epochs'] = self.num_epochs

        loss_g = None
        for epoch in range(1, self.num_epochs):
            self.status['epoch'] = epoch
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                image = doc[self.input_field]
                targets = doc[self.target_field]
                loss = self._train(image, targets)
                loss_g = 0.99 * loss_g + 0.01 * float(loss) if loss_g is not None else float(loss)
                lr = self.optimizer.param_groups[0]['lr']

                self.status['loop'] += 1
                self.status['loss'] = loss
                self.status['loss_g'] = loss_g
                self.status['lr'] = lr

                info = f'[{epoch}/{self.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)

            self.status['progress'] = self.status['epoch'] / self.num_epochs

            if callable(self.epoch_callback):
                self.epoch_callback(self)


