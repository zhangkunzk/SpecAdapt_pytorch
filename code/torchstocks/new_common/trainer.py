#!/usr/bin/env python3

import abc
import collections
import re
from copy import deepcopy
from typing import Union, Sequence, Mapping, Type, Tuple, Callable

from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from torchstocks import dist
from torchstocks import optim as _optim
from torchstocks.optim import LRScheduler

__all__ = [
    'init_model',
    'parse_param_groups',
    'init_optimizer',
    'init_train_dataset',
    'init_lr_scheduler',
    'AbstractTrainer'
]


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


def init_model(model, param_groups, device):
    model.to(device)

    if param_groups:
        mode_spec = {}
        for group in param_groups:
            if 'mode' in group:
                mode = group['mode']
                if not isinstance(mode, bool):
                    mode = mode == 'train'
                if 'name' in group:
                    mode_spec[group['name']] = mode
                if 'names' in group:
                    for name in group['names']:
                        mode_spec[name] = mode
        model.train = ModeHandler(model, mode_spec)

    model = dist.convert_model(model)
    return model


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


def init_optimizer(
        optimizer: Union[str, Type[optim.Optimizer]],
        model: nn.Module,
        param_groups,
        **opt_args  # e.g., lr, momentum, weight_decay, ...
) -> optim.Optimizer:
    # get optimizer constructor
    if isinstance(optimizer, str):
        try:
            OptimizerType = getattr(_optim, optimizer)
        except AttributeError:
            OptimizerType = getattr(optim, optimizer)
    else:
        OptimizerType = optimizer

    # parse args
    opt_args['params'] = parse_param_groups(param_groups, model)
    if 'momentum' in opt_args:
        opt_args['betas'] = (opt_args['momentum'], 0.999)  # for Adam*

    # create optimizer instance
    co = OptimizerType.__init__.__code__
    optimizer = OptimizerType(**{
        name: opt_args[name]
        for name in co.co_varnames[1:co.co_argcount]
        if name in opt_args
    })
    return optimizer


def init_train_dataset(
        dataset: Dataset,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=None,
        transform=None
) -> Tuple[Dataset, DataLoader, Callable]:
    if dataset is not None:
        if not (hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')):
            raise RuntimeError(f'Invalid dataset {type(dataset)}.')

        if transform is None:
            if hasattr(dataset, 'transform') and callable(dataset.transform):
                transform = dataset.transform

        if collate_fn is None:
            if hasattr(dataset, 'collate_fn') and callable(dataset.collate_fn):
                collate_fn = dataset.collate_fn

        dataset = dist.convert_dataset(dataset, is_train=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    ) if dataset is not None and len(dataset) > 0 else None

    return dataset, loader, transform


def init_lr_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: Union[str, Type],
        num_loops: int,
        min_value: float = 0
) -> LRScheduler:
    if isinstance(scheduler_type, str):
        SchedulerType = getattr(_optim, scheduler_type)
    else:
        SchedulerType = scheduler_type
    return LRScheduler(optimizer, SchedulerType(num_loops, min_value=min_value))


class AbstractTrainer(abc.ABC):

    def __init__(self):
        self._status = {}

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

    init_model = staticmethod(init_model)
    parse_param_groups = staticmethod(parse_param_groups)
    init_optimizer = staticmethod(init_optimizer)
    init_train_dataset = staticmethod(init_train_dataset)
    init_lr_scheduler = staticmethod(init_lr_scheduler)
