#!/usr/bin/env python3


from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.nn.memory import CosineDistance
from torchstocks.common.trainer import BPTrainerWithDataset


class ContrastiveTrainer(BPTrainerWithDataset):

    def __init__(
            self,
            model: nn.Module,
            unlabeled_dataset: Dataset,
            train_dataset: Dataset,
            test_dataset: Dataset,
            unlabeled_collate=None,
            train_collate=None,
            test_collate=None,
            input_field: str = 'feature',
            label_field: str = 'label',
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            eval_interval: int = 5,
            device: str = 'cpu',
            eval_method: str = 'centroid'
    ) -> None:
        super(ContrastiveTrainer, self).__init__(
            model=model,
            train_dataset=unlabeled_dataset,
            train_collate=unlabeled_collate,
            auxiliary_dataset=[train_dataset, test_dataset] if train_dataset and test_dataset else None,
            auxiliary_collate=[train_collate, test_collate],
            optimizer=optimizer,
            batch_size=batch_size,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            drop_last=True,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            device=device
        )
        self.input_field = input_field
        self.label_field = label_field
        self.eval_interval = eval_interval

        self.unlabeled_loader = self.train_loader
        if self.auxiliary_loader:
            self.train_loader = self.auxiliary_loader[0]
            self.test_loader = self.auxiliary_loader[1]
        else:
            self.train_loader = None
            self.test_loader = None
        self.eval_method = eval_method

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            h = self.model(x)
            return h.detach().cpu()

    def train_step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        loss = self.model(x1, x2)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.detach().cpu()

    def train(self):
        if self.unlabeled_loader is None:
            return

        loss_g = None
        for epoch in range(self.num_epochs):
            self.set_status('epoch', epoch + 1)

            self.model.train()
            loop = tqdm(self.unlabeled_loader, leave=False, ncols=96)
            for batch_idx, doc in enumerate(loop):
                self.set_status('loop', epoch * len(loop) + batch_idx)

                x1, x2 = doc[self.input_field]
                loss = self.train_step(x1, x2)

                loss = float(loss)
                loss_g = 0.9 * loss_g + 0.1 * loss if loss_g is not None else loss
                lr = self.optimizer.param_groups[0]['lr']
                self.set_status('loss', loss)
                self.set_status('loss_g', loss_g)
                self.set_status('lr', lr)
                loop.set_description(f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)

            print_string = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f}'
            if (epoch + 1) % self.eval_interval == 0:
                self.evaluate()
                if 'metrics' in self.status:
                    for k, v in self.get_status('metrics').items():
                        print_string += f' {k}={v:.04f}'
            print(print_string)

    def evaluate(self):
        self.del_status('metrics')
        if self.train_loader is None or self.test_loader is None:
            return

        self.model.eval()

        if self.eval_method == 'logistic':
            # get train embeddings
            feature_list, label_list = [], []
            loop = tqdm(self.train_loader, leave=False, desc='Testing', ncols=96)
            for doc in loop:
                feature = self.predict_step(doc[self.input_field]).numpy()
                label = doc[self.label_field].numpy()
                feature_list.extend(feature)
                label_list.extend(label)
            train_feature = np.array(feature_list)
            train_label = np.array(label_list)

            # get test embeddings
            feature_list, label_list = [], []
            loop = tqdm(self.test_loader, leave=False, desc='Testing', ncols=96)
            for doc in loop:
                feature = self.predict_step(doc[self.input_field]).numpy()
                label = doc[self.label_field].numpy()
                feature_list.extend(feature)
                label_list.extend(label)
            test_feature = np.array(feature_list)
            test_label = np.array(label_list)

            # normalize the features
            mean = np.mean(train_feature, 0, keepdims=True)
            sigma = np.sqrt(np.var(train_feature, 0, keepdims=True))
            train_feature = (train_feature - mean) / (sigma + 1e-10)
            test_feature = (test_feature - mean) / (sigma + 1e-10)

            # perform the classification through LR
            classifier = LogisticRegression(max_iter=10000)
            classifier.fit(train_feature, train_label)
            pred_label = classifier.predict(test_feature)
            acc = accuracy_score(test_label, pred_label)
            self.set_status('metrics', {'accuracy': acc})
        elif self.eval_method == 'centroid':
            dummy_input = torch.rand((1, 3, 256, 256), dtype=torch.float32)
            dims = self.predict_step(dummy_input).shape[-1]
            proto_dict = defaultdict(list)
            loop = tqdm(self.train_loader, leave=False, desc='Testing', ncols=96)
            for doc in loop:
                feature = self.predict_step(doc[self.input_field])
                label = doc[self.label_field]
                for i in range(label.shape[0]):
                    class_index = int(label[i])
                    if class_index not in proto_dict:
                        proto_dict[class_index].extend([torch.zeros(size=(dims,), dtype=torch.float32), 0])
                    else:
                        proto_dict[class_index][0] += feature[i]
                        proto_dict[class_index][1] += 1
            proto_list = []
            for i in range(len(proto_dict)):
                proto_list.append(proto_dict[i][0] / proto_dict[i][1])
            train_proto = torch.stack(proto_list, dim=0)  # (k, d)

            iteration = 0
            acc = 0.
            dist_fn = CosineDistance()
            loop = tqdm(self.test_loader, leave=False, desc='Testing', ncols=96)
            for doc in loop:
                feature = self.predict_step(doc[self.input_field])
                label = doc[self.label_field]
                dist = dist_fn(feature, train_proto)  # [n_test, k]
                pred_label = torch.argmin(dist, dim=1)  # [n_test, ]
                acc += accuracy_score(label.numpy(), pred_label.numpy())
                iteration += 1
            acc /= iteration
            self.set_status('metrics', {'accuracy': acc})
        else:
            raise ValueError('Unsupported eval method.')
