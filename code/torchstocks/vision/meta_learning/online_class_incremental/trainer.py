#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-10-31
"""

import math
import collections
from typing import Sequence, MutableMapping, MutableSequence, Union
from copy import deepcopy

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchstocks.optim import LRScheduler, CosineWarmupDecay
from torchstocks.utils.metrics import ClassificationMeter
from torchstocks.common.trainer import AbstractTrainer

__all__ = [
    'OnlineClassIncrementalTrainer'
]


class OnlineClassIncrementalTrainer(AbstractTrainer):

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            # outer
            optimizer: str = 'AdamW',
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            # inner
            test_optimizer: str = 'Adam',
            test_inner_lr: float = 5e-4,
            test_momentum: float = 0,
            test_weight_decay: float = 0,
            # overall
            batch_size: int = 1,
            num_epochs: int = 100,
            test_repeat_num: int = 10,
            num_workers: int = 10,
            clip_grad_norm: float = 0.1,
            lr_scheduler: bool = False,
            device: str = 'cpu',
            input_field: str = 'image',
            target_field: str = 'label',
            output_dir: str = None,
            # param related
            param_groups: list = None,
            inner_updated_layers: Sequence[str] = ('head',),

    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.optimizer: Union[str, optim.Optimizer] = optimizer
        self.max_lr = max_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.test_optimizer: Union[str, optim.Optimizer] = test_optimizer
        self.test_inner_lr = test_inner_lr
        self.test_momentum = test_momentum
        self.test_weight_decay = test_weight_decay

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.test_repeat_num = test_repeat_num
        self.num_workers = num_workers
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.input_field = input_field
        self.target_field = target_field
        self.output_dir = output_dir

        self.param_groups: MutableSequence[MutableMapping] = deepcopy(param_groups)
        self.inner_updated_layers = inner_updated_layers

        self.inner_updated_param_list = []
        self.status = {}

        self._init_model()
        self._init_dataloader()
        self._init_param_groups()
        self._init_optimizer()
        if self.lr_scheduler:
            self._init_scheduler()

    def _init_model(self):
        self.model.to(self.device)

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        ) if self.train_dataset is not None else None

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        ) if self.test_dataset is not None else None

    def _init_param_groups(self):
        # outer param
        if self.param_groups is None:
            self.param_groups = []
        # collect group names
        group_names = []
        for group in self.param_groups:
            if 'name' in group:
                assert isinstance(group['name'], str)
                group_names.append(group['name'])
            if 'names' in group:
                assert isinstance(group['names'], (list, tuple))
                group_names.extend(group['names'])
        group_names.sort(reverse=True)

        # assign parameters to groups and assign inner-updated parameters to list
        group_assign = collections.defaultdict(list)
        default_group = []
        for name, param in self.model.network.named_parameters():
            for group_name in group_names:
                if name.startswith(group_name):
                    group_assign[group_name].append(param)
                    break
            else:
                default_group.append(param)
            for layer in self.inner_updated_layers:
                if name.startswith(layer):
                    self.inner_updated_param_list.append(param)
        # add parameters to param_groups
        if default_group:
            self.param_groups.append({'params': default_group})

        for group in self.param_groups:
            if 'lr' in group and (group['lr'] is None or group['lr'] == 0):
                group.clear()
            if 'params' not in group:
                group['params'] = []
            params = group['params']
            if 'name' in group:
                assert isinstance(group['name'], str)
                params.extend(group_assign[group['name']])
            if 'names' in group:
                assert isinstance(group['names'], (list, tuple))
                for name in group['names']:
                    params.extend(group_assign[name])
        self.param_groups = [group for group in self.param_groups if group]

    def _init_optimizer(self):
        # Create optimizer.
        opt_args = {
            'params': self.param_groups,
            'lr': self.max_lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,  # SGD, RMSprop
            'betas': (self.momentum, 0.999),  # Adam*
        }
        OptimizerType = getattr(optim, self.optimizer)
        co = OptimizerType.__init__.__code__
        self.optimizer = OptimizerType(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })
        # Create test optimizer.
        test_opt_args = {
            'params': self.inner_updated_param_list,
            'lr': self.test_inner_lr,
            'weight_decay': self.test_weight_decay,
            'momentum': self.test_momentum,  # SGD, RMSprop
            'betas': (self.test_momentum, 0.999),  # Adam*
        }
        TestOptimizerType = getattr(optim, self.test_optimizer)
        test_co = TestOptimizerType.__init__.__code__
        self.test_optimizer = TestOptimizerType(**{
            name: test_opt_args[name]
            for name in test_co.co_varnames[1:test_co.co_argcount]
            if name in test_opt_args
        })

        if self.clip_grad_norm is not None:
            original_step = self.optimizer.step

            def step_wrapper():
                params = []
                for group in self.optimizer.param_groups:
                    params.extend(group['params'])
                clip_grad_norm_(params, self.clip_grad_norm, math.inf)
                original_step()
            self.optimizer.step = step_wrapper

    def _init_scheduler(self):
        self.num_loops = self.num_epochs * len(self.train_loader)
        self.scheduler = LRScheduler(self.optimizer, CosineWarmupDecay(self.num_loops))

    def train_step(self, sx, sy, qx, qy):
        sx = sx.to(self.device)
        sy = sy.to(self.device)
        qx = qx.to(self.device)
        qy = qy.to(self.device)
        loss = self.model(sx, sy, qx, qy)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.lr_scheduler:
            self.scheduler.step()
        return loss.detach().cpu()

    def predict_step(self, support_x, support_y, query_x):
        """Predict a sequential task. Each task is consist of several samples
        Args:
            support_x (torch.Tensor): (1, train_samples, ...)
            support_y (torch.Tensor): (1, train_samples, ...)
            query_x (torch.Tensor): (1, test_samples, ...)
        Returns:
            torch.Tensor: query_y (test_samples, ...)
        """
        self.model.checkpoint()
        for layer_name in self.inner_updated_layers:
            layer = getattr(self.model.network, layer_name)
            nn.init.kaiming_normal_(layer.weight)

        pred_q = []
        for sx, sy, qx in zip(support_x, support_y, query_x):  # only one iteration
            steps = sx.shape[0]  # the all number of meta-test-train images
            with tqdm(total=steps, leave=False, ncols=64) as loop:
                for i in range(steps):
                    loop.update()
                    train_x = sx[i].unsqueeze(0).to(self.device)
                    train_y = sy[i].unsqueeze(0).to(self.device)
                    pred = self.model.network(train_x)
                    self.test_optimizer.zero_grad()
                    loss = self.model.criterion(pred, train_y)
                    loss.backward()
                    self.test_optimizer.step()
                    loop.set_description(
                        f'meta-test-training '
                        f'step={i} '
                        f'loss={loss.item():.06f}',
                        False
                    )
            # to evaluate fast
            test_batch = 16
            iterations = int(qx.shape[0] / test_batch)
            remainder = qx.shape[0] % test_batch
            if remainder != 0:
                iterations += 1

            with torch.no_grad():
                self.model.eval()
                for i in range(iterations):
                    if remainder != 0 and i == iterations - 1:
                        end = qx.shape[0]
                        start = end - remainder
                    else:
                        start = i * test_batch
                        end = start + test_batch
                    test_x = qx[start:end].to(self.device)
                    pred = self.model.network(test_x)
                    pred = torch.argmax(pred, 1)
                    pred_q.append(pred)
            self.model.train()

        self.model.restore()
        return torch.cat(pred_q).detach().cpu()

    def train(self):
        if self.train_loader is None:
            return
        self.status['loop'] = 0
        loss_g = None
        all_loops = self.num_epochs * len(self.train_loader)

        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch + 1
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, ncols=74)
            for query_doc, supp_doc in loop:
                loss = self.train_step(
                    sx=supp_doc[self.input_field],
                    sy=supp_doc[self.target_field],
                    qx=query_doc[self.input_field],
                    qy=query_doc[self.target_field]
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
                    if 'metrics' in self.status:
                        print_string = ''
                        print_string += f' after {self.test_repeat_num} runs,'
                        for k, v in self.status['metrics'].items():
                            print_string += f' {k}={v:.04f}'
                        print(print_string)

    def evaluate(self):
        if self.test_loader is None:
            return

        meter = ClassificationMeter()
        cur_num = 0
        for query_doc, supp_doc in self.test_loader:
            if cur_num >= self.test_repeat_num:
                break
            pred_y = self.predict_step(
                support_x=supp_doc[self.input_field],
                support_y=supp_doc[self.target_field],
                query_x=query_doc[self.input_field]
            )
            meter.update(output=pred_y.numpy(), target=query_doc[self.target_field].numpy())
            cur_num += 1
        self.status['metrics'] = {
            'Acc': meter.accuracy(),
            'F1': meter.f1().mean()
        }
