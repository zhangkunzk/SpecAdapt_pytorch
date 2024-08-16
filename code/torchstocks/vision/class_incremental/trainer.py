#!/usr/bin/env python3

"""
@author: xi Howie
@since: 2022-07-18
"""

import collections
from typing import Union, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from torchstocks.optim import LRScheduler, CosineWarmupDecay
from torchstocks.vision.class_incremental.evaluate import ClassificationMeter

__all__ = [
    'Trainer'
]


def freeze_module(module: nn.Module):
    module.freeze = True
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze_module(module: nn.Module):
    module.freeze = False
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def switch_to_train(module: nn.Module):
    module.train()
    _check_freeze(module)


def _check_freeze(module: nn.Module):
    if hasattr(module, 'freeze') and module.freeze:
        module.eval()
    else:
        for child in module.children():
            _check_freeze(child)


def switch_to_eval(module: nn.Module):
    module.eval()


def add_class(classifier: nn.Module, new_num_classes: int):
    if isinstance(classifier, nn.Linear):
        old_weight = classifier.weight.data
        if classifier.bias is not None:
            is_bias = True
        else:
            is_bias = False
        device = next(classifier.parameters()).device
        old_out_features, in_features = old_weight.shape
        num_added = new_num_classes - old_out_features
        assert num_added >= 0, "Incremental classes should not less than original classes"
        new_classifier = nn.Linear(in_features=in_features, out_features=new_num_classes, device=device, bias=False)
        nn.init.kaiming_normal_(new_classifier.weight, mode='fan_in', nonlinearity='relu')
        new_classifier.weight.data[:old_out_features, :] = old_weight
        if is_bias:
            old_bias = classifier.bias.data
            new_classifier.bias.data[:old_out_features] = old_bias
        return new_classifier, num_added

    elif isinstance(classifier, nn.Conv2d):
        old_weight = classifier.weight.data
        if classifier.bias is not None:
            is_bias = True
        else:
            is_bias = False
        device = next(classifier.parameters()).device
        stride, padding, kernel = classifier.stride, classifier.padding, classifier.kernel_size
        old_out_features, in_features = old_weight.shape[0], old_weight.shape[1]
        num_added = new_num_classes - old_out_features
        new_classifier = nn.Conv2d(in_channels=in_features, out_channels=new_num_classes, stride=stride,
                                   padding=padding, kernel_size=kernel, bias=is_bias, device=device)
        new_classifier.weight.data[:old_out_features] = old_weight
        if is_bias:
            old_bias = classifier.bias.data
            new_classifier.bias.data[:old_out_features] = old_bias
        return new_classifier, num_added

    else:
        raise RuntimeError(f'Unsupported classifier {type(classifier)}.')


def extract_classifier(model: nn.Module):
    name_list = [name for name, value in model.state_dict().items()]
    param_name = name_list[-1]
    layer_name = param_name.split('.')[:-1]
    layer_name = '.'.join(layer_name)
    classifier = getattr(model, layer_name)
    return classifier, layer_name


class Trainer(object):

    def __init__(
            self,
            model: nn.Module,
            train_dataset,
            test_dataset,
            rehearsal_dataset,
            num_class,
            batch_size,
            device
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.rehearsal_dataset = rehearsal_dataset
        self.num_class = num_class

        self.optimizer_name = 'AdamW'
        self.max_lr = 1e-3
        self.min_lr = 0.0
        self.weight_decay = 0.3
        self.momentum = 0.9
        self.batch_size = batch_size
        self.num_epochs = 100
        self.output_dir = None

        self.param_groups = None
        self.optimizer = None
        self.lr_scheduler = None
        self.epoch = None
        self.loss = None

        self.model.to(self.device)
        self._init_dataloader()

    def _init_dataloader(self):
        if len(self.rehearsal_dataset) != 0:
            self.train_loader = DataLoader(
                ConcatDataset([self.train_dataset, self.rehearsal_dataset]),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=10,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=10,
                pin_memory=True,
                persistent_workers=True
            )
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=10,
                pin_memory=True
            )

    def _init_param_groups(self):
        default_params = []
        grouped_params = collections.defaultdict(list)
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, 'group'):
                grouped_params[p.group].append(p)
            else:
                default_params.append(p)
        self.param_groups = [{'params': default_params}]
        for group, params in grouped_params.items():
            # todo: different lr and weight_decay for different groups
            self.param_groups.append({'params': params})

    def _init_optimizer(self):
        from torch import optim
        opt_class = getattr(optim, self.optimizer_name, None)
        if opt_class is None:
            from torchstocks import optim
            opt_class = getattr(optim, self.optimizer_name, None)
        assert opt_class is not None
        opt_args = {
            'params': self.param_groups,
            'lr': self.max_lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,  # SGD, RMSprop
            'betas': (self.momentum, 0.999),  # Adam*
            'nesterov': True  # SGD
        }
        co = opt_class.__init__.__code__
        self.optimizer = opt_class(**{
            name: opt_args[name]
            for name in co.co_varnames[1:co.co_argcount]
            if name in opt_args
        })
        self.lr_scheduler = LRScheduler(
            self.optimizer,
            CosineWarmupDecay(
                self.num_epochs * len(self.train_loader),
                min_value=self.min_lr / self.max_lr
            )
        )
        self.epoch = None
        self.loss = None

    def train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        loss = self._compute_loss(y_, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        loss = loss.detach().cpu()
        return loss

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(target, output.shape[-1]).float()
        output = F.softmax(output, -1)
        loss = -((output * target).sum(-1) + 1e-10).log().mean()
        return loss

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            y = output.argmax(-1)
            y = y.detach().cpu()
            return y

    def run_train(self):
        self._init_param_groups()
        self._init_optimizer()

        for self.epoch in range(self.num_epochs):
            switch_to_train(self.model)
            self._train_epoch()
            self.run_test()

    def _train_epoch(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train(doc['image'], doc['label'])
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def run_test(self):
        msg = ''
        if self.epoch is not None:
            msg += f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f}'
            meter = ClassificationMeter(self.num_class)
            # print(self.num_class)
            switch_to_eval(self.model)
            progress = tqdm(total=len(self.test_loader), leave=False, ncols=96)
            for query_doc in self.test_loader:
                progress.update()
                y_pred = self.predict(query_doc['image'])
                meter.update(output=y_pred.numpy(), target=query_doc['label'].numpy())
            progress.close()
            average_acc, total_list, tp_list = meter.average_accuracy()
            acc = meter.accuracy()
            msg += f' average ACC =  {acc:.02%}'
            msg += f' total list =  {total_list}'
            msg += f' true list =  {tp_list}'
        print(msg)

    def transfer_model(self, num_class):
        pass
