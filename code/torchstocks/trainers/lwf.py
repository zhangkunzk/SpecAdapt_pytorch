#!/usr/bin/env python3
"""
Since: 2022/11/1
Author: Howie
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .incremental import IncrementalTrainer
from .incremental import add_class, extract_classifier

__all__ = [
    'LwFTrainer'
]


def freeze_old_model(model: nn.Module):
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    return old_model


class LwFTrainer(IncrementalTrainer):
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 test_dataset,
                 rehearsal_dataset,
                 num_class,
                 device,
                 _lambda=1
                 ):
        model.to(device)
        self.old_model = freeze_old_model(model)
        classifier, classifier_name = extract_classifier(model)
        new_classifier, self.num_added = add_class(classifier, num_class)
        model.__dict__[classifier_name] = new_classifier

        super(LwFTrainer, self).__init__(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            rehearsal_dataset=rehearsal_dataset,
            num_class=num_class,
            device=device,
        )
        self._lambda = _lambda

    def train_incremental(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_old = self.old_model(x)
        old_class_num = y_old.shape[1]
        loss_new = self._compute_weak_loss(y_, y, old_class_num)
        loss_old = self._compute_old_loss(y_, y_old)
        loss = loss_new + self._lambda * loss_old
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def train_warmup(self, x, y, warmup_class):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_ = y_[:, -warmup_class:]
        y = y - warmup_class
        loss = self._compute_loss(y_, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def _compute_old_loss(self, output, target, t=2, eps=1e-10) -> torch.Tensor:
        """
        output: (n, new_classes)
        target: (n, old_class)
        """
        old_class = target.shape[1]
        output = output[:, :old_class]  # (n, old_class)
        output = F.softmax(output, dim=1)
        target = F.softmax(target, dim=1)
        output = (output + eps).pow(1 / t)
        output = output / (output.sum(1, keepdim=True) + eps)
        target = (target + eps).pow(1 / t)
        target = target / (target.sum(1, keepdim=True) + eps)
        loss = target * torch.log(output + eps)  # (n, old_class)
        loss = - (loss.sum(1).mean())
        return loss

    def _compute_weak_loss(self, output: torch.Tensor, target: torch.Tensor, old_class_num: int) -> torch.Tensor:
        target = target - old_class_num
        output = output[:, old_class_num:]
        target = F.one_hot(target, output.shape[-1]).float()
        output = F.softmax(output, -1)
        loss = -((output * target).sum(-1) + 1e-10).log().mean()
        return loss

    def train(self):
        self.status['epoch'] = 0
        if self.num_added == 0:
            for epoch in range(self.num_epochs):
                self.status['epoch'] = epoch
                self.model.train()
                self._train_epoch()
                if (epoch + 1) % 10 == 0:
                    self.evaluate()
        else:
            for epoch in range(self.num_epochs):
                self.status['epoch'] = epoch
                self.model.train()
                self._train_epoch_incremental()
                if (epoch + 1) % 10 == 0:
                    self.evaluate()

    def _train_epoch_incremental(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_incremental(doc[self.input_field], doc[self.target_field])
            self.status['loss'] = 0.9 * self.status['loss'] + 0.1 * float(loss) \
                if self.status['loss'] != 0 else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.status["epoch"] + 1}/{self.num_epochs}] L={self.status["loss"]:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()
