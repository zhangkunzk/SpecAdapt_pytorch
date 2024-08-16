#!/usr/bin/env python3

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.new_common.trainer import AbstractTrainer
from torchstocks.optim import LinearWarmupCosineDecay

__all__ = [
    'ClassificationTrainer'
]


class ClassificationTrainer(AbstractTrainer):
    """Classification trainer
    """

    def __init__(
            self,
            #
            # dataset options
            model: nn.Module,
            train_dataset: Dataset,
            batch_size: int = 256,
            num_workers: int = 10,
            shuffle=True,
            input_field: str = 'image',
            target_field: str = 'label',
            #
            # optimizer options
            optimizer: str = 'AdamW',
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            clip_grad_norm: float = None,
            param_groups: list = None,
            num_epochs: int = 100,
            opt_args=None,
            #
            # misc
            device: str = 'cpu',
            use_amp: bool = False,
            epoch_callback=None,
    ) -> None:
        super(ClassificationTrainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.input_field = input_field
        self.target_field = target_field

        self.optimizer = optimizer
        self.max_lr = max_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.param_groups = param_groups
        self.num_epochs = num_epochs
        self.opt_args = opt_args if opt_args is not None else {}

        self.device = device
        self.use_amp = use_amp
        self.epoch_callback = epoch_callback

        self.model = self.init_model(self.model, self.param_groups, self.device)
        self.train_dataset, self.train_loader, _ = self.init_train_dataset(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle
        )
        if self.train_loader:
            self.num_loops = len(self.train_loader) * self.num_epochs
            self.optimizer = self.init_optimizer(
                optimizer=self.optimizer,
                model=self.model,
                param_groups=self.param_groups,
                lr=self.max_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                clip_grad_norm=self.clip_grad_norm,
                **self.opt_args
            )
            self.lr_scheduler = self.init_lr_scheduler(self.optimizer, LinearWarmupCosineDecay, self.num_loops)

        self.status['progress'] = 0
        self.scaler = amp.GradScaler()

    def _train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        if self.use_amp:
            with amp.autocast():
                loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.update()
            self.optimizer.step()
            if 'scale' not in self.status or self.scaler._scale != self.status['scale']:
                self.status['scale'] = self.scaler._scale
                print('scale changed', self.scaler._scale)
        else:
            loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        return loss.detach().cpu()

    def run(self):
        if self.train_loader is None:
            return

        self.status['loop'] = 0
        self.status['num_epochs'] = self.num_epochs
        loss_g = None
        for epoch in range(1, self.num_epochs + 1):
            self.status['epoch'] = epoch
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                x, y = doc[self.input_field], doc[self.target_field]
                loss = float(self._train(x, y))
                if loss_g is None:
                    loss_g = loss
                loss_g = 0.9 * loss_g + 0.1 * loss
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
