#!/usr/bin/env python3
"""
Since: 2022/11/7
Author: Howie
"""
import collections
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import torchstocks.vision.class_incremental.trainer as trainer
from torchstocks.vision.class_incremental.trainer import switch_to_eval
from torchstocks.vision.class_incremental.dataset import TrainDataset
from torchstocks.optim import LRScheduler, CosineWarmupDecay
from torchstocks.vision.class_incremental.trainer import add_class, extract_classifier
from torchstocks.vision.class_incremental.evaluate import ClassificationMeter
from torchstocks.vision.class_incremental.model import BaseModel

__all__ = [
    'AlgTrainer'
]


def freeze_model(model):
    new_model = copy.deepcopy(model)
    new_model.eval()
    for p in new_model.parameters():
        p.required_grad = False
    return new_model


def distill(output, target, t=3, eps=1e-10):
    output = F.softmax(output, dim=1)
    target = F.softmax(target, dim=1)
    output = (output + eps).pow(1 / t)
    output = output / (output.sum(1, keepdim=True) + eps)
    target = (target + eps).pow(1 / t)
    target = target / (target.sum(1, keepdim=True) + eps)
    loss = target * torch.log(output + eps)  # (n, class)
    return - (loss.sum(1).mean())


def distill_softmax_t(output, target, t=3, eps=1e-10):
    output = F.softmax((output / t), dim=1)
    target = F.softmax((target / t), dim=1)
    loss = target * torch.log(output + eps)  # (n, class)
    return - (loss.sum(1).mean())


def _compute_weak_loss(output: torch.Tensor, target: torch.Tensor, old_class_num: int) -> torch.Tensor:
    target = target - old_class_num
    output = output[:, old_class_num:]
    target = F.one_hot(target, output.shape[-1]).float()
    output = F.softmax(output, -1)
    loss = -((output * target).sum(-1) + 1e-10).log().mean()
    return loss


def _compute_distill_loss(output, target_old, target_new):
    """
    output: (n, new_classes)
    target_old: (n, old_class)
    target_new: (n, new_classes)
    """
    old_class = target_old.shape[1]
    output_old = output[:, :old_class]  # (n, old_class)
    output_new = output[:, old_class:]  # (n, new_classes - old_classes)
    target_new = target_new[:, old_class:]  # (n, new_classes - old_classes)

    loss_old = distill_softmax_t(output_old, target_old)
    loss_new = distill_softmax_t(output_new, target_new)
    return loss_old, loss_new


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
            kd_percentage=0.25,
            ce_weight=0.0001
    ):
        super().__init__(model, train_dataset, test_dataset, rehearsal_dataset, num_class, batch_size,
                         device)
        self.samples_size = 20
        self.kd_percentage = kd_percentage
        self.ce_weight = ce_weight
        self.old_model = freeze_model(model)
        self.kd_weight = 1 - self.ce_weight
        self._model_restructure()
        print(self.num_class_added)
        self.new_percent = self.num_class_added / self.num_class

    def _model_restructure(self):
        classifier, classifier_name = extract_classifier(self.model)
        new_classifier, self.num_class_added = add_class(classifier, self.num_class)
        if self.num_class_added > 0:
            self.model.__dict__[classifier_name] = new_classifier
            self.new_model = copy.deepcopy(self.model)
            for p in self.new_model.backbone.layers[0].parameters():
                p.required_grad = False
            for p in self.new_model.backbone.layers[1].parameters():
                p.required_grad = False
            for p in self.new_model.backbone.layers[2].parameters():
                p.required_grad = False

    def _init_param_groups(self, model):
        default_params = []
        grouped_params = collections.defaultdict(list)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, 'group'):
                grouped_params[p.group].append(p)
            else:
                default_params.append(p)
        self.param_groups = [{'params': default_params}]
        for group, params in grouped_params.items():
            self.param_groups.append({'params': params})

    def rebuild_scheduler(self, num_epochs):
        self.lr_scheduler = LRScheduler(
            self.optimizer,
            CosineWarmupDecay(
                num_epochs * len(self.train_loader),
                min_value=self.min_lr / self.max_lr
            )
        )

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(target, output.shape[-1]).float()
        output = F.softmax(output, -1)
        loss = target * torch.log(output + 1e-10) + (1 - target) * torch.log((1 - output) + 1e-10)
        loss = - (loss.sum(1).mean())
        return loss

    def run_train(self):
        kd_epochs = int(self.kd_percentage * self.num_epochs)
        if self.num_class_added == 0:
            self._init_param_groups(self.model)
            self._init_optimizer()
            self._balance_dataset()
            for self.epoch in range(self.num_epochs):
                self.model.train()
                self._train_epoch()
                if (self.epoch + 1) % 50 == 0:
                    self.run_test()

        else:
            self._init_param_groups(self.new_model)
            self._init_optimizer()
            self.rebuild_scheduler(kd_epochs)
            for self.epoch in range(kd_epochs):
                self.new_model.train()
                self.new_model.backbone.layers[0].eval()
                self.new_model.backbone.layers[1].eval()
                self.new_model.backbone.layers[2].eval()
                self._train_stage1_epoch()
                if (self.epoch + 1) % 50 == 0:
                    self.run_test_new_model()
            self._prepare_fine_tune()
            self.rebuild_scheduler((self.num_epochs - kd_epochs))
            for self.epoch in range(kd_epochs, self.num_epochs):
                self.model.train()
                self._train_stage2_epoch()
                # self.maintaining_fairness()
                if (self.epoch + 1) % 50 == 0:
                    self.run_test()
            # self.rebuild_scheduler(100)
            # self.ce_weight = 0.5
            # self.new_percent /= 2
            # self.kd_weight = 0.5
            # nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_in')
            # for self.epoch in range(self.num_epochs, self.num_epochs + 100):
            #     self.model.train()
            #     self._train_stage3_epoch()
            #     # self.maintaining_fairness()
            #     if (self.epoch + 1) % 50 == 0:
            #         self.run_test()

    def _train_stage1_epoch(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_stage1(doc['image'], doc['label'])
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def train_stage1(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.new_model(x)
        y_old = self.old_model(x)
        old_class_num = y_old.shape[-1]
        '''only train new classifier by new data'''
        loss_distill = distill_softmax_t(y_[:, :old_class_num], y_old)
        loss = 0.8 * self._compute_loss(y_, y) + 0.2 * loss_distill
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def _train_stage2_epoch(self):
        progress = tqdm(total=len(self.train_loader), leave=False, ncols=96)
        for doc in self.train_loader:
            progress.update()
            loss = self.train_stage2(doc['image'], doc['label'])
            self.loss = 0.9 * self.loss + 0.1 * float(loss) if self.loss is not None else float(loss)
            lr = self.optimizer.param_groups[0]['lr']
            info = f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f} LR={lr:.02e}'
            progress.set_description(info, False)
        progress.close()

    def train_stage2(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)
        y_ = self.model(x)
        y_new = self.new_model(x)
        y_old = self.old_model(x)
        '''distill old model and new model'''
        loss_ce = self._compute_loss(y_, y)
        loss_distill_o, loss_distill_n = _compute_distill_loss(y_, y_old, y_new)
        loss = ((self.kd_weight - self.new_percent) * loss_distill_o +
                self.new_percent * loss_distill_n +
                self.ce_weight * self.new_percent * loss_ce)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        loss = loss.detach().cpu()
        return loss

    def _train_stage3_epoch(self):
        progress = tqdm(total=len(self.fine_tune_loader), leave=False, ncols=96)
        for doc in self.fine_tune_loader:
            progress.update()
            loss = self.train_stage2(doc['image'], doc['label'])
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
            image_size=64
        )

        self.fine_tune_loader = DataLoader(
            self.fine_tune_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )

    def _prepare_fine_tune(self):
        self._balance_dataset()
        self._create_fine_tune_loader()
        for p in self.new_model.parameters():
            p.requires_grad = False
        self.new_model.eval()
        for p, p_old, p_new in zip(self.model.backbone.parameters(), self.old_model.backbone.parameters(),
                                   self.new_model.backbone.parameters()):
            p.data[:] = (p_old.data[:] + p_new.data[:]) / 2
        self.weight_decay = 0.01
        self._init_param_groups(self.model)
        self._init_optimizer()

    def run_test_new_model(self):
        msg = ''
        if self.epoch is not None:
            msg += f'[{self.epoch + 1}/{self.num_epochs}] L={self.loss:.06f}'
            meter = ClassificationMeter(self.num_class)
            switch_to_eval(self.new_model)
            progress = tqdm(total=len(self.test_loader), leave=False, ncols=96)
            for query_doc in self.test_loader:
                progress.update()
                y_pred = self.predict_new(query_doc['image'])
                meter.update(output=y_pred.numpy(), target=query_doc['label'].numpy())
            progress.close()
            # average_acc, total_list, tp_list = meter.average_accuracy()
            acc = meter.accuracy()
            # msg += f' tasks ACC =  {np.around(average_acc, 2)}'
            msg += f' average ACC =  {acc:.02%}'
            # msg += f' predict =  {total_list}'
            # msg += f' tp =  {tp_list}'
        print(msg)

    def predict_new(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            output = self.new_model(x)
            y = output.argmax(-1)
            y = y.detach().cpu()
            return y

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            new_mean = output[:, -self.num_class_added:].mean()
            old_mean = output[:, :-self.num_class_added].mean()
            # print('new mean: ', new_mean)
            # print('old mean: ', old_mean)
            y = output.argmax(-1)
            y = y.detach().cpu()
            # output_ = output.cpu().numpy()
            return y

    def maintaining_fairness(self):
        new_classes_num = self.num_class_added
        weight = self.model.fc.weight.data
        old_classes_weights = weight[:-new_classes_num, :].norm(p=2, dim=1).mean()
        new_classes_weights = weight[-new_classes_num:, :].norm(p=2, dim=1).mean()
        gamma = old_classes_weights / new_classes_weights
        weight[-new_classes_num:, :] *= gamma
        # weight_norm = weight.norm(p=2, dim=1)
        # weight /= (weight_norm.unsqueeze(-1))
