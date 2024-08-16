#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-11-11
"""

import os

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.common.trainer import BPTrainerWithDataset
from torchstocks.utils.metrics import ClassificationMeter

__all__ = [
    'FewShotTrainer'
]


class FewShotTrainer(BPTrainerWithDataset):

    def __init__(
            self,
            image_size: int,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            optimizer: str = 'AdamW',
            batch_size: int = 4,
            max_lr: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 4,
            num_workers: int = 10,
            drop_last: bool = False,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            device: str = 'cpu',
            input_field: str = 'image',
            target_field: str = 'label',
            output_dir: str = None
    ) -> None:
        super().__init__(
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
            drop_last=drop_last,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            device=device
        )
        self.test_loader = self.auxiliary_loader
        self.image_size = image_size
        self.input_field = input_field
        self.target_field = target_field
        self.output_dir = output_dir

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

    def train_step(self, support_x, support_y, query_x, query_y):
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)

        loss = self.model(support_x, support_y, query_x, query_y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.detach().cpu()

    def predict_step(self, support_x, support_y, query_x):
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)

        qy_list = []
        self.model.checkpoint()
        for sx, sy, qx in zip(support_x, support_y, query_x):
            for i in range(self.model.num_steps * 2):
                pred = self.model.network(sx)
                loss = self.model.criterion(pred, sy)
                loss.backward()
                with torch.no_grad():
                    for p in self.model.network.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        p.add_(p.grad, alpha=-self.model.inner_lr)
                        p.grad = None

            pred = self.model.network(qx)
            qy = torch.argmax(pred, 1)
            qy_list.append(qy)
            self.model.restore()
        return torch.stack(qy_list).detach().cpu()

    def train(self):
        if self.train_loader is None:
            return
        self.status['loop'] = 0
        loss_g = None
        self.model.train()
        all_loops = self.num_epochs * len(self.train_loader)

        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch + 1
            loop = tqdm(self.train_loader, leave=False, ncols=64)
            for query_doc, supp_doc in loop:
                loss = self.train_step(
                    support_x=supp_doc[self.input_field],
                    support_y=supp_doc[self.target_field],
                    query_x=query_doc[self.input_field],
                    query_y=query_doc[self.target_field]
                )
                if loss_g is None:
                    loss_g = loss
                loss_g = 0.9 * loss_g + 0.1 * loss
                lr = self.optimizer.param_groups[0]['lr']

                self.status['loop'] += 1
                self.status['loss'] = loss
                self.status['loss_g'] = loss_g
                self.status['lr'] = lr

                cur_loop = self.status['loop']
                info = f'[{cur_loop}/{all_loops}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)
                if cur_loop % 300 == 0 or cur_loop == all_loops:
                    self.evaluate()

    def evaluate(self):
        if self.test_loader is None:
            return

        meter = ClassificationMeter()
        loop = tqdm(self.test_loader, ncols=64, leave=False)
        for query_doc, supp_doc in loop:
            pred_y = self.predict_step(
                support_x=supp_doc[self.input_field],
                support_y=supp_doc[self.target_field],
                query_x=query_doc[self.input_field]
            )
            meter.update(output=pred_y.numpy(), target=query_doc[self.target_field])

        self.status['metrics'] = {
            'Acc': meter.accuracy(),
            'F1': meter.f1().mean()
        }

        print_string = ''
        if self.has_status('loop') and self.has_status('loss_g'):
            cur_loop = self.get_status('loop')
            loss_g = self.get_status('loss_g')
            print_string += f'[{cur_loop}/{self.num_epochs * len(self.train_loader)}] L={loss_g:.06f}'
        for k, v in self.status['metrics'].items():
            print_string += f' {k}={v:.04f}'
        print(print_string)
