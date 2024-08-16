#!/usr/bin/env python3

import abc
import collections
import re
from copy import deepcopy
from typing import Union, MutableMapping, MutableSequence, Sequence, Mapping

from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from torchstocks import dist
from torchstocks import optim as _optim
from torchstocks.common.dataset import DataCollate
from torchstocks.optim import LRScheduler

__all__ = [
    'AbstractTrainer',
    'BPTrainer',
    'BPTrainerWithDataset'
]


class AbstractTrainer(abc.ABC):

    def __init__(self):
        self._status = {}

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
        self._status[name] = value

    def get_status(self, name: str):
        return self._status[name]

    def del_status(self, name: str):
        if name in self._status:
            del self._status[name]

    def has_status(self, name: str) -> bool:
        return name in self._status

    @property
    def status(self):
        return self._status


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


def parse_param_groups(
        param_groups: Sequence[Mapping],
        model: nn.Module,
        verbose=False
) -> Sequence[Mapping]:
    # Copy the groups first, since the group will be edited later.
    tmp_groups = (
        deepcopy(param_groups)
        if param_groups is not None
        else []
    )

    # Collect rules to match the groups.
    all_rule_keys = {
        'prefix', 'name', 'prefixes', 'names',  # match by prefix
        'match'  # match by regular expression(s)
    }
    all_rules = []
    for group in tmp_groups:
        rules = []
        rule_keys = list(all_rule_keys & group.keys())
        if len(rule_keys) == 0:
            continue
        elif len(rule_keys) != 1:
            pass
        else:
            rule_key = rule_keys[0]
            rule_or_rules = group[rule_key]
            if isinstance(rule_or_rules, str):
                if rule_key == 'match':
                    rule_or_rules = 'REGEX:' + rule_or_rules
                rules.append(rule_or_rules)
            elif isinstance(rule_or_rules, (list, tuple)):
                if rule_key == 'match':
                    rule_or_rules = ['REGEX:' + rule for rule in rule_or_rules]
                rules.extend(rule_or_rules)
            else:
                raise RuntimeError(f'Unsupported {rule_key} type {type(rule_or_rules)}.')
        group['rule'] = rules
        all_rules.extend(rules)

    # Assign parameters to groups by the matching rules.
    # The matching order is determined by the original defined order.
    # The parameter will be assigned to the LAST matched rule.
    group_assign = collections.defaultdict(list)
    default_group = []
    for name, param in model.named_parameters():
        matched_rule = None
        for rule in all_rules:
            if rule.startswith('REGEX:'):
                if re.search(rule[len('REGEX:'):], name):
                    matched_rule = rule
            else:
                if name.startswith(rule):
                    matched_rule = rule
        if matched_rule is not None:
            group_assign[matched_rule].append(param)
            if verbose:
                print(f'Assign [{name}]  ->  [{matched_rule}].')
        else:
            default_group.append(param)
            if verbose:
                print(f'Assign [{name}]  ->  [default group].')

    # add parameters to param_groups
    if default_group:
        tmp_groups.append({'params': default_group})

    output_groups = []
    for group in tmp_groups:
        if 'lr' in group and (group['lr'] is None or group['lr'] == 0):
            continue
        if 'rule' in group:
            if 'params' not in group:
                group['params'] = []
            params = group['params']
            for rule in group['rule']:
                params.extend(group_assign[rule])
            del group['rule']
        output_groups.append(group)
    return output_groups


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
            clip_grad_norm: float = None,
            lr_scheduler: str = 'LinearWarmupCosineDecay',
            lr_decay_min_value: float = 0,
            device: str = 'cpu',
            rank: Union[int, float] = 0.5,
            update_rank_rate: float = 0.5
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
        self.scheduler: Union[str, LRScheduler] = lr_scheduler
        self.min_value = lr_decay_min_value
        self.device = dist.convert_device(device)
        self.rank = rank  # low rank
        self.update_rank_rate = update_rank_rate
        self._init_model()
        if self.num_loops:
            self.param_groups = parse_param_groups(self.param_groups, self.model)
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

        self.model = dist.convert_model(self.model)

    def _init_optimizer(self):
        # Create optimizer.
        opt_args = {
            'params': self.param_groups,
            'lr': self.max_lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,  # SGD, RMSprop
            'betas': (self.momentum, 0.999),  # Adam*
            'r': self.rank,  # peft, like lora ...
            'update_rank_rate': self.update_rank_rate,
            'clip_grad_norm': self.clip_grad_norm
        }
        try:
            OptimizerType = getattr(_optim, self.optimizer)
        except AttributeError:
            OptimizerType = getattr(optim, self.optimizer)
        co = OptimizerType.__init__.__code__
        self.optimizer = OptimizerType(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })

    def _init_scheduler(self):
        self.scheduler = LRScheduler(
            self.optimizer, getattr(_optim, self.scheduler)(self.num_loops, min_value=self.min_value)
        )


class BPTrainerWithDataset(BPTrainer, abc.ABC):

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            auxiliary_dataset: Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]],
            train_collate: DataCollate = None,
            auxiliary_collate: Union[DataCollate, Sequence[DataCollate], Mapping[str, DataCollate]] = None,
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            shuffle=True,
            drop_last: bool = False,
            param_groups: list = None,
            clip_grad_norm: float = None,
            lr_scheduler: str = 'LinearWarmupCosineDecay',
            lr_decay_min_value: float = 0,
            device: str = 'cpu',
            rank: Union[int, float] = 0.5,
            update_rank_rate: float = 0.5
    ) -> None:
        self.train_dataset = dist.convert_dataset(train_dataset, is_train=True)
        self.auxiliary_dataset = dist.convert_dataset(auxiliary_dataset, is_train=False)
        self.train_collate = train_collate
        self.auxiliary_collate = auxiliary_collate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.shuffle = shuffle
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
            lr_scheduler=lr_scheduler,
            lr_decay_min_value=lr_decay_min_value,
            device=device,
            rank=rank,
            update_rank_rate=update_rank_rate
        )

    def _init_train_dataset(self):
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                drop_last=self.drop_last,
                collate_fn=self.train_collate
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
                        pin_memory=True,
                        collate_fn=_collate
                    )
                    for _dataset, _collate in zip(self.auxiliary_dataset, self.auxiliary_collate)
                ]
            elif isinstance(self.auxiliary_dataset, dict):
                self.auxiliary_loader = {
                    name: DataLoader(
                        _dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers // 2,
                        pin_memory=True,
                        collate_fn=(
                            self.auxiliary_collate.get(name)
                            if isinstance(self.auxiliary_collate, dict)
                            else None
                        )
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
                    pin_memory=True,
                    collate_fn=self.auxiliary_collate
                )
        else:
            self.auxiliary_loader = None
