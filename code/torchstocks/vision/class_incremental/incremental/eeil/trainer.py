#!/usr/bin/env python3
"""
Since: 2022/8/22
Author: Howie
"""
import copy
import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchstocks.vision.class_incremental.trainer as trainer
from torchstocks.vision.class_incremental.dataset import TrainDataset
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
            _lambda=1
    ):
        super().__init__(model, train_dataset, test_dataset, rehearsal_dataset, num_class, batch_size,
                         device)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.samples_size = 20
        self.num_class_added = None
        self._lambda = _lambda

    def train_incremental(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_old = self.old_model(x)
        loss_new = self._compute_loss(y_, y)
        loss_old = self._compute_old_distill_loss(y_, y_old)
        loss = loss_new + self._lambda * loss_old
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def _compute_old_distill_loss(self, output, target, t=2, eps=1e-10) -> torch.Tensor:
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

    def run_train(self):
        unfreeze_module(self.model)
        classifier, classifier_name = extract_classifier(self.model)
        new_classifier, num_added = add_class(classifier, self.num_class)
        self.num_class_added = num_added
        if self.num_class_added == 0:
            self._init_param_groups()
            self._init_optimizer()
            for self.epoch in range(self.num_epochs):
                switch_to_train(self.model)
                self._train_epoch()
                if (self.epoch + 1) % 10 == 0:
                    self.run_test()
            self._balance_dataset()
        else:
            stage_one_epochs = int(self.num_epochs * 0.7)
            self.model.__dict__[classifier_name] = new_classifier
            self._init_param_groups()
            self._init_optimizer()
            for self.epoch in range(stage_one_epochs):
                switch_to_train(self.model)
                self._train_epoch_incremental()
                if (self.epoch + 1) % 10 == 0:
                    self.run_test()
            self._prepare_fine_tune()
            for self.epoch in range(stage_one_epochs, self.num_epochs):
                switch_to_train(self.model)
                self._train_epoch_fine_tune()
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

    def _balance_dataset(self):
        self.rehearsal_dataset.dataset.write_fixed_size(self.train_dataset.dataset, self.num_class)

    def _create_fine_tune_loader(self):
        self.fine_tune_dataset = TrainDataset(
            self.rehearsal_dataset.dataset.write_path,
            image_size=32
        )
        self.fine_tune_loader = DataLoader(
            self.fine_tune_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True
        )

    def _train_epoch_fine_tune(self):
        progress = tqdm(total=len(self.fine_tune_loader), leave=False, ncols=96)
        for doc in self.fine_tune_loader:
            progress.update()
            loss = self.train_fine_tune(doc['image'], doc['label'])
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def train_fine_tune(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_learned = self.old_model(x)
        loss_cls = self._compute_loss(y_, y)
        loss_distill = self._compute_new_distill_loss(y_, y_learned)
        loss = loss_cls + self._lambda * loss_distill
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def _compute_new_distill_loss(self, output, target, t=2, eps=1e-10):
        output = output[:, -self.num_class_added:]  # (n, added_classes)
        target = target[:, -self.num_class_added:]  # (n, added_classes)
        output = F.softmax(output, dim=1)
        target = F.softmax(target, dim=1)
        output = (output + eps).pow(1 / t)
        output = output / (output.sum(1, keepdim=True) + eps)
        target = (target + eps).pow(1 / t)
        target = target / (target.sum(1, keepdim=True) + eps)
        loss = target * torch.log(output + eps)  # (n, added_classes)
        loss = - (loss.sum(1).mean())
        return loss

    def _prepare_fine_tune(self):
        self._balance_dataset()
        self._create_fine_tune_loader()
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.max_lr = self.max_lr / 10
        self._init_optimizer()
