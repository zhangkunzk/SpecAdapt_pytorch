#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-14
@last editor: Yubin
@last time: 2022-11-11
"""

import math
import copy
from typing import Literal

import torch
from torch import autograd
from torch import nn


class Model(nn.Module):

    def __init__(
            self,
            network: nn.Module, *,
            criterion: Literal['CrossEntropyLoss'] = 'CrossEntropyLoss',
            inner_lr: float = 0.01,
            num_steps: int = 5,
            first_order=False
    ) -> None:
        super(Model, self).__init__()
        self.network = network
        if criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid Loss "{criterion}".')
        assert inner_lr > 0
        self.inner_lr = inner_lr

        assert num_steps >= 1
        self.num_steps = num_steps

        self.first_order = first_order

        memo = {id(p): p for p in network.parameters()}
        self._network_symbol = copy.deepcopy(network, memo)

        self._param_spec = {}
        self._make_param_spec(self._network_symbol)
        self._param_list = list(self.network.parameters())

        self._state_dict = None

    def _make_param_spec(self, module):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, nn.Parameter):
                delattr(module, name)
                self._param_spec[id(obj)] = (module, name)
        for child in module.children():
            self._make_param_spec(child)

    def forward(self, support_x, support_y, query_x, query_y):
        return torch.mean(torch.stack([
            self._per_task(sx, sy, qx, qy)
            for sx, sy, qx, qy in zip(support_x, support_y, query_x, query_y)
        ]))

    def _per_task(self, support_x, support_y, query_x, query_y):
        param_list = self._param_list
        for i in range(self.num_steps):
            pred_y = self.network(support_x) if i == 0 else self._network_symbol(support_x)
            loss = self.criterion(pred_y, support_y)
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
                new_param = param_list[j] - self.inner_lr * grad_list[j]
                new_param_list.append(new_param)
                module, name = self._param_spec[id(self._param_list[j])]
                setattr(module, name, new_param)
            param_list = new_param_list

        pred_y = self._network_symbol(query_x)
        loss = self.criterion(pred_y, query_y)
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

    def __init__(self, in_channels, out_channels, batch_norm=True, non_linear=True):
        super(Layer, self).__init__(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True) if non_linear else nn.Identity(),
            nn.MaxPool2d((2, 2), (2, 2)),
        )


class ConvNet(nn.Module):

    def __init__(self, image_size, num_class, ch_hid=64):
        super().__init__()
        self.layer1 = Layer(3, ch_hid)
        image_size = math.floor(image_size / 2.0)
        self.layer2 = Layer(ch_hid, ch_hid)
        image_size = math.floor(image_size / 2.0)
        self.layer3 = Layer(ch_hid, ch_hid)
        image_size = math.floor(image_size / 2.0)
        self.layer4 = Layer(ch_hid, ch_hid)
        image_size = math.floor(image_size / 2.0)
        self._flat_size = image_size * image_size * ch_hid

        self.fc = nn.Linear(self._flat_size, num_class)

    def forward(self, x: torch.Tensor):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = h.reshape(-1, self._flat_size)

        h = self.fc(h)
        return h
