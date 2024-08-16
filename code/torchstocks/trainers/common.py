#!/usr/bin/env python3

import abc
import collections
import math
from copy import deepcopy
from typing import Union, MutableMapping, MutableSequence, Sequence, Mapping

from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from torchstocks.optim import LRScheduler, CosineWarmupDecay

__all__ = [
    'AbstractTrainer',
    'BPTrainer',
    'BPTrainerWithDataset'
]


class AbstractTrainer(abc.ABC):

    def __init__(self):
        self.status = {}

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    def set_status(self, name: str, value):
        self.status[name] = value

    def get_status(self, name: str):
        return self.status[name]

    def del_status(self, name: str):
        if name in self.status:
            del self.status[name]

    def has_status(self, name: str) -> bool:
        return name in self.status


class ModeHandler(object):

    def __init__(self, module: nn.Module, mode_spec):
        self.module = module
        self.original_handler = module.train
        self.mode_spec = mode_spec

    def __call__(self, mode=True):
        ret = self.original_handler(mode)
        for name, target_mode in self.mode_spec.items():
            if mode == target_mode:
                continue
            self.module.get_submodule(name).train(target_mode)
        return ret


class BPTrainer(AbstractTrainer, abc.ABC):

    def __init__(
            self,
            model: nn.Module,
            num_loops: int = None,
            optimizer: str = 'AdamW',
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            device: str = 'cpu'
    ) -> None:
        super(BPTrainer, self).__init__()
        self.model: nn.Module = model
        self.num_loops = num_loops
        self.optimizer: Union[str, optim.Optimizer] = optimizer
        self.max_lr = max_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.param_groups: MutableSequence[MutableMapping] = deepcopy(param_groups)
        self.clip_grad_norm = clip_grad_norm
        self.device = device

        self._init_model()
        if self.num_loops:
            self._init_param_groups()
            self._init_optimizer()
            self._init_scheduler()

    def _init_model(self):
        self.model.to(self.device)

        if self.param_groups is None:
            self.param_groups = []

        mode_spec = {}
        for group in self.param_groups:
            if 'mode' in group:
                mode = group['mode']
                if not isinstance(mode, bool):
                    mode = mode == 'train'
                if 'name' in group:
                    mode_spec[group['name']] = mode
                if 'names' in group:
                    for name in group['names']:
                        mode_spec[name] = mode
        self.model.train = ModeHandler(self.model, mode_spec)

    def _init_param_groups(self):
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

        # assign parameters to groups
        group_assign = collections.defaultdict(list)
        default_group = []
        for name, param in self.model.named_parameters():
            for group_name in group_names:
                if name.startswith(group_name):
                    group_assign[group_name].append(param)
                    break
            else:
                default_group.append(param)

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

        if self.clip_grad_norm is not None:
            original_step = self.optimizer.step

            def step_wrapper():
                params = []
                for group in self.optimizer.param_groups:
                    params.extend(group['params'])
                clip_grad_norm_(params, self.clip_grad_norm, math.inf)
                original_step()

            self.optimizer.step = step_wrapper

        # for group in self.optimizer.param_groups:
        #     group['params'] = len(group['params'])
        #     print(group)
        # exit()

    def _init_scheduler(self):
        self.scheduler = LRScheduler(self.optimizer, CosineWarmupDecay(self.num_loops))


class BPTrainerWithDataset(BPTrainer, abc.ABC):

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            auxiliary_dataset: Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]],
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            drop_last: bool = False,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            device: str = 'cpu'
    ) -> None:
        self.train_dataset = train_dataset
        self.auxiliary_dataset = auxiliary_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.drop_last = drop_last

        self._init_train_dataset()
        self._init_test_dataset()

        super(BPTrainerWithDataset, self).__init__(
            model=model,
            num_loops=self.num_loops,
            optimizer=optimizer,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            device=device
        )

    def _init_train_dataset(self):
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                drop_last=self.drop_last
            )
            self.num_loops = self.num_epochs * len(self.train_loader)
        else:
            self.train_loader = None
            self.num_loops = None

    def _init_test_dataset(self):
        if self.auxiliary_dataset:
            if isinstance(self.auxiliary_dataset, (tuple, list)):
                self.auxiliary_loader = [
                    DataLoader(
                        _dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers // 2,
                        pin_memory=True
                    )
                    for _dataset in self.auxiliary_dataset
                ]
            elif isinstance(self.auxiliary_dataset, dict):
                self.auxiliary_loader = {
                    name: DataLoader(
                        _dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers // 2,
                        pin_memory=True
                    )
                    for name, _dataset in self.auxiliary_dataset.items()
                }
            else:
                if not (hasattr(self.auxiliary_dataset, '__len__') and hasattr(self.auxiliary_dataset, '__getitem__')):
                    raise RuntimeError(f'Invalid test_dataset {type(self.auxiliary_dataset)}.')
                self.auxiliary_loader = DataLoader(
                    self.auxiliary_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers // 2,
                    pin_memory=True
                )
        else:
            self.auxiliary_loader = None
