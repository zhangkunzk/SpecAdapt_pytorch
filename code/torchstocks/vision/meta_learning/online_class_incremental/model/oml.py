#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-11-1
@reference: Meta-Learning Representations for Continual Learning
@code reference: Guangyi's maml-pytorch
"""

import copy
from typing import Literal, List, Sequence

import torch
from torch import autograd
from torch import nn
import numpy as np


def inner_update_param(model: nn.Module, layers: List):
    inner_updated_params = []
    for name, param in model.named_parameters():
        for layer in layers:
            if name.startswith(layer):
                inner_updated_params.append(param)
    return inner_updated_params


def reset_head(layer, index):
    ''' To facilitate the propagation of gradients through the model we prevent memorization of
        training examples by randomizing the weights in the last fully connected layer corresponding
        to the task that is about to be learned
        reference: https://github.com/uvm-neurobotics-lab/higherANML/: anml.py
    '''
    for i in index:
        nn.init.kaiming_normal_(layer.weight[i].unsqueeze(0))


class Model(nn.Module):

    def __init__(
            self,
            network: nn.Module,
            inner_lr: float = 0.1,
            criterion: Literal['CrossEntropyLoss'] = 'CrossEntropyLoss',
            first_order: bool = False,
            inner_update_layers: Sequence[str] = ('head',),  # the layer that will be updated in inner loop
            head_name: str = 'head'
    ) -> None:
        super(Model, self).__init__()
        self.network = network
        if criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid Loss "{criterion}".')
        assert inner_lr > 0
        self._inner_lr = inner_lr

        self.first_order = first_order
        self.inner_update_layers = inner_update_layers
        memo = {id(p): p for p in self.network.parameters()}
        self._network_symbol = copy.deepcopy(self.network, memo)
        self._param_spec = {}
        self._init_inner_param()
        self._param_list = inner_update_param(network, self.inner_update_layers)
        self._head_layer = getattr(network, head_name)
        self._state_dict = None

    def _init_inner_param(self):
        for name, child_module in self._network_symbol.named_children():
            for layer in self.inner_update_layers:
                if name.startswith(layer):
                    self._make_param_spec(child_module)

    def _make_param_spec(self, module):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, nn.Parameter):
                delattr(module, name)
                self._param_spec[id(obj)] = (module, name)
        for child_module in module.children():
            self._make_param_spec(child_module)

    def forward(self, support_x, support_y, query_x, query_y):
        '''
        meta-train:
        support_x: a sequential images, shape:[Batch, Samples_Supp, C, H, W]
        query_x: images of remember set, shape:[Batch, Samples_Query, C, H, W]
        '''
        loss = []
        for sx, sy, qx, qy in zip(support_x, support_y, query_x, query_y):
            loss.append(self._per_task(sx, sy, qx, qy))
        return torch.mean(torch.stack(loss))

    def _per_task(self, batch_x_train, batch_y_train, batch_x_test, batch_y_test):
        param_list = self._param_list
        reset_label_list = np.unique(batch_y_train.cpu().numpy())
        reset_head(self._head_layer, reset_label_list)

        num_steps = batch_x_train.shape[0]
        for i in range(num_steps):

            x_train = batch_x_train[i].unsqueeze(0)  # (1, C, H, W)
            y_train = batch_y_train[i].unsqueeze(0)  # (1, 1)

            if i == 0:
                pred_y = self.network(x_train)
            else:
                pred_y = self._network_symbol(x_train)

            loss = self.criterion(pred_y, y_train)
            if len(loss.shape) != 0:
                loss = loss.mean()
            new_param_list = []
            if self.first_order:
                grad_list = [
                    g.detach()
                    for g in autograd.grad(loss, param_list)
                ]
            else:
                grad_list = autograd.grad(loss, param_list, create_graph=True)

            for j in range(len(param_list)):
                new_param = param_list[j] - self._inner_lr * grad_list[j]
                new_param_list.append(new_param)
                module, name = self._param_spec[id(self._param_list[j])]
                setattr(module, name, new_param)
            param_list = new_param_list

        pred_y = self._network_symbol(batch_x_test)
        loss = self.criterion(pred_y, batch_y_test)
        if len(loss.shape) != 0:
            loss = loss.mean()
        return loss

    def checkpoint(self):
        self._state_dict = {
            name: value.clone().detach()
            for name, value in self.network.state_dict().items()
        }

    def restore(self):
        self.network.load_state_dict(self._state_dict)


class Layer(nn.Sequential):

    def __init__(self, in_channels, out_channels, batch_norm=True, non_linear=True, pooling=False):
        super(Layer, self).__init__(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True) if non_linear else nn.Identity(),
            nn.MaxPool2d((2, 2), (2, 2)) if pooling else nn.Identity(),
        )


class Backbone(nn.Module):

    def __init__(self, image_size, ch_hid=256, out_dim=1024):
        super(Backbone, self).__init__()
        self.layer1 = Layer(3, ch_hid)
        self.layer2 = Layer(ch_hid, ch_hid, pooling=True)
        self.layer3 = Layer(ch_hid, ch_hid)
        self.layer4 = Layer(ch_hid, ch_hid, pooling=True)
        self.layer5 = Layer(ch_hid, ch_hid)
        self.layer6 = Layer(ch_hid, ch_hid, pooling=True)
        image_size = int(image_size/8)
        self._flat_size = image_size * image_size * ch_hid
        self.fc = nn.Linear(self._flat_size, out_dim)

    def forward(self, x: torch.Tensor):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = h.reshape(-1, self._flat_size)
        h = self.fc(h)
        return h


class ConvNetwork(nn.Module):

    def __init__(self, image_size, num_classes, ch_hid=256, out_dim=1024) -> None:
        super(ConvNetwork, self).__init__()
        self.backbone = Backbone(image_size, ch_hid, out_dim)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
