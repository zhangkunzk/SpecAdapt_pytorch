#!/usr/bin/env python3
"""
Since: 2022/11/1
Author: Howie
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm

from .common import BPTrainer
from ..utils.metrics.class_incremental import IncrementalMeter

__all__ = [
    'IncrementalTrainer',
    'add_class',
    'extract_classifier'
]


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
        new_classifier = nn.Linear(
            in_features=in_features,
            out_features=new_num_classes,
            bias=is_bias,
            device=device
        )
        new_classifier.weight.data[:old_out_features] = old_weight
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
        new_classifier = nn.Conv2d(
            in_channels=in_features,
            out_channels=new_num_classes,
            stride=stride,
            padding=padding,
            kernel_size=kernel,
            bias=is_bias,
            device=device
        )
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


class IncrementalTrainer(BPTrainer):
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 test_dataset,
                 rehearsal_dataset,
                 num_class,
                 optimizer: str = 'AdamW',
                 batch_size: int = 256,
                 max_lr: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 0.3,
                 num_epochs: int = 100,
                 num_workers: int = 10,
                 param_groups: list = None,
                 clip_grad_norm: float = 0.1,
                 input_field: str = 'image',
                 target_field: str = 'label',
                 device: str = 'cpu'
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.rehearsal_dataset = rehearsal_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.num_class = num_class
        self._init_train_dataset()
        self._init_test_dataset()
        self.input_field = input_field
        self.target_field = target_field

        super(IncrementalTrainer, self).__init__(
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
        self.status['loss'] = 0

    def _init_train_dataset(self):
        if len(self.rehearsal_dataset) != 0:
            self.train_loader = DataLoader(
                ConcatDataset([self.train_dataset, self.rehearsal_dataset]),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
        self.num_loops = self.num_epochs * len(self.train_loader)

    def _init_test_dataset(self):
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        loss = self._compute_loss(y_, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss = loss.detach().cpu()
        return loss

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(target, output.shape[-1]).float()
        output = F.softmax(output, -1)
        loss = -((output * target).sum(-1) + 1e-10).log().mean()
        return loss

    def predict_step(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            y = output.argmax(-1)
            y = y.detach().cpu()
            return y

    def train(self):
        self.status['epoch'] = 0
        for self.epoch in range(self.num_epochs):
            self.model.train()
            self._train_epoch()
            self.evaluate()

    def _train_epoch(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_step(doc[self.input_field], doc[self.target_field])
            self.status['loss'] = 0.9 * self.status['loss'] + 0.1 * float(loss) \
                if self.status['loss'] != 0 else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.status["epoch"] + 1}/{self.num_epochs}] L={self.status["loss"]:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def evaluate(self):
        msg = ''
        msg += f'[{self.status["epoch"]}/{self.num_epochs}] L={self.status["loss"]:.06f}'
        meter = IncrementalMeter(self.num_class)
        self.model.eval()
        progress = tqdm(total=len(self.test_loader), leave=False, ncols=96)
        for query_doc in self.test_loader:
            progress.update()
            y_pred = self.predict_step(query_doc[self.input_field])
            meter.update(output=y_pred.numpy(), target=query_doc[self.target_field].numpy())
        progress.close()
        average_acc, total_list, tp_list = meter.average_accuracy()
        msg += f' tasks ACC =  {np.around(average_acc, 2)}'
        msg += f' average ACC =  {np.mean(average_acc):.02%}'
        msg += f' predict =  {total_list}'
        msg += f' tp =  {tp_list}'
        print(msg)
