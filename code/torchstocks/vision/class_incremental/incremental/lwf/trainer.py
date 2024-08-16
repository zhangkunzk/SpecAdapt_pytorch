#!/usr/bin/env python3
"""
Since: 2022/8/8
Author: Howie
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torchstocks.vision.class_incremental.trainer as trainer
from torchstocks.vision.class_incremental.trainer import add_class, switch_to_train, extract_classifier, freeze_module, \
    unfreeze_module

__all__ = [
    'AlgTrainer'
]


class AlgTrainer(trainer.Trainer):
    def __init__(
            self,
            model: nn.Module,
            train_dataset,
            test_dataset,
            rehearsal_dataset,
            num_class,
            batch_size,
            device,
            _lamda=1
    ):
        super().__init__(model, train_dataset, test_dataset, rehearsal_dataset, num_class, batch_size, device)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self._lamda = _lamda

    def train_incremental(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_old = self.old_model(x)
        old_class_num = y_old.shape[1]
        loss_new = self._compute_weak_loss(y_, y, old_class_num)
        # loss_old = self._compute_old_loss(y_, y_old)
        loss_old = self.softmax_t(y_, y_old)
        # l1_loss = self.l1_norm()
        loss = (1 - self._lamda) * loss_new + self._lamda * loss_old
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
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
        self.lr_scheduler.step()
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

    def softmax_t(self, output, target, t=2, eps=1e-10) -> torch.Tensor:
        """
        output: (n, new_classes)
        target: (n, old_class)
        """
        old_class = target.shape[1]
        output = output[:, :old_class]  # (n, old_class)
        output = F.softmax(output / t, dim=1)
        target = F.softmax(target / t, dim=1)
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

    def l1_norm(self, alpha=0.2):
        l1_loss = torch.abs(self.model.weight).sum()
        return -alpha * l1_loss

    def run_train(self):
        classifier, classifier_name = extract_classifier(self.model)
        new_classifier, num_added = add_class(classifier, self.num_class)

        if num_added == 0:
            self._init_param_groups()
            self._init_optimizer()
            for self.epoch in range(self.num_epochs):
                switch_to_train(self.model)
                self._train_epoch()
                if (self.epoch + 1) % 10 == 0:
                    self.run_test()
        else:
            self._lamda = (self.num_class - num_added) / self.num_class
            self.model.__dict__[classifier_name] = new_classifier
            self._init_param_groups()
            self._init_optimizer()
            for self.epoch in range(self.num_epochs):
                switch_to_train(self.model)
                self._train_epoch_incremental()
                if (self.epoch + 1) % 10 == 0:
                    self.run_test()

    def _train_epoch_incremental(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_incremental(doc['image'], doc['label'])
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def _train_warmup(self, warmup_class):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        freeze_module(self.model.backbone)
        unfreeze_module(self.model.fc)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_warmup(doc['image'], doc['label'], warmup_class)
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        unfreeze_module(self.model.backbone)
        progress.close()
