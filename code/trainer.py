#!/usr/bin/env python3

import os

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.common.trainer import BPTrainerWithDataset
from torchstocks.utils.metrics import ClassificationMeter

__all__ = [
    'ClassificationTrainer'
]


class ClassificationTrainer(BPTrainerWithDataset):

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            shuffle=True,
            drop_last=False,
            param_groups: list = None,
            clip_grad_norm: float = None,
            lr_decay_min_value: float = 1e-6,
            device: str = 'cpu',
            input_field: str = 'image',
            target_field: str = 'label',
            output_dir: str = None,
            output_file: str = None,
            eval_interval: int = 10,
            rank: float = None,
            largest_singulars: bool = False
    ) -> None:
        super(ClassificationTrainer, self).__init__(
            model=model,
            train_dataset=train_dataset,
            auxiliary_dataset=test_dataset,
            optimizer=optimizer,
            batch_size=batch_size,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            lr_decay_min_value=lr_decay_min_value,
            device=device,
            rank=rank
        )
        self.test_loader = self.auxiliary_loader
        self.input_field = input_field
        self.target_field = target_field
        self.output_dir = output_dir
        self.output_file = output_file
        self.eval_interval = eval_interval

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        loss = self.model(x, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.detach().cpu()

    def predict_step(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            y_ = self.model(x)
            return y_.detach().cpu()

    def train(self):
        if self.train_loader is None:
            return

        self.status['loop'] = 0
        loss_g = None
        last_acc = 0.0
        for epoch in range(1, self.num_epochs + 1):
            self.status['epoch'] = epoch
            self.model.train()
            self.optimizer.train()
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for doc in loop:
                x, y = doc[self.input_field], doc[self.target_field]
                loss = float(self.train_step(x, y))
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
            if epoch % self.eval_interval == 0 or epoch == self.num_epochs:
                self.evaluate()
                if 'metrics' in self.status:
                    if self.output_dir is not None:
                        last_file = f'last_{self.output_file}' if self.output_file else 'last.pth'
                        torch.save(self.model, os.path.join(self.output_dir, last_file))
                        if self.status['metrics']['Acc'] and self.status['metrics']['Acc'] > last_acc:
                            best_file = f'best_{self.output_file}' if self.output_file else 'best.pth'
                            torch.save(self.model, os.path.join(self.output_dir, best_file))

    def evaluate(self):
        if self.test_loader is None:
            return

        meter = ClassificationMeter()
        self.model.eval()
        self.optimizer.eval()
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            x, y = doc[self.input_field], doc[self.target_field]
            y_ = self.predict_step(x)
            meter.update(output=y_.numpy(), target=y.numpy())

        self.status['metrics'] = {
            'Acc': meter.accuracy(),
            'F1': meter.f1().mean()
        }

        if self.has_status('epoch') and self.has_status('loss_g'):
            epoch = self.get_status('epoch')
            loss_g = self.get_status('loss_g')
            print(f'[{epoch}/{self.num_epochs}] L={loss_g:.06f}', end='')
        for k, v in self.status['metrics'].items():
            print(f' {k}={v:.04f}', end='')
        print()
