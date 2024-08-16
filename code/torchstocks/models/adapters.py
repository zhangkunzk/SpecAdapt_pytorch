#!/usr/bin/env python3

"""
@author: xi
@since: 2022-04-11
"""

import math
from typing import Sequence

import torch
from einops import rearrange
from torch import nn


class _Hook(object):

    def __init__(self):
        self.x = None

    def __call__(self, m, x, y):
        self.x = x[0]
        return y


class ResnetAdapter(nn.Module):

    def __init__(self, model: nn.Module, before_relu=False) -> None:
        super(ResnetAdapter, self).__init__()
        layer0_list = [model.conv1, model.bn1]
        if hasattr(model, 'maxpool'):
            layer0_list.append(getattr(model, 'maxpool'))
        if hasattr(model, 'relu'):
            layer0_list.append(getattr(model, 'relu'))
        if hasattr(model, 'relu1'):
            layer0_list.append(getattr(model, 'relu1'))
        if hasattr(model, 'act'):
            layer0_list.append(getattr(model, 'act'))
        if hasattr(model, 'act1'):
            layer0_list.append(getattr(model, 'act1'))

        self.layers = nn.ModuleList([
            nn.Sequential(*layer0_list),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        ])

        if before_relu:
            self.hooks = [_Hook() for _ in range(5)]
            relu = self.layers[0][-1]
            relu.inplace = False
            relu.register_forward_hook(self.hooks[0])
            for i in range(1, 5):
                relu = self.layers[i][-1].relu
                relu.inplace = False
                relu.register_forward_hook(self.hooks[i])
        else:
            self.hooks = None

        # the following code will inference the net to get feature size
        # setting the backbone to eval mode to prevent its BN layers from being corrupted
        self.ch_out_list = []
        self.stride_list = []
        state = self.training
        self.train(False)
        for output in self(torch.rand((1, 3, 512, 512), dtype=torch.float32)):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            _, c, h, w = output.shape
            assert h == w
            self.ch_out_list.append(int(c))
            self.stride_list.append(math.ceil(512 / h))
        self.train(state)

    def forward(self, x: torch.Tensor, depth: int = None) -> Sequence[torch.Tensor]:
        if depth is None:
            depth = len(self.layers)
        ys = []
        y = x
        for i in range(depth):
            y = self.layers[i](y)
            ys.append(self.hooks[i].x if self.hooks is not None else y)
        return ys


class ViTAdapter(nn.Module):

    def __init__(self, vit: nn.Module):
        super(ViTAdapter, self).__init__()
        self.vit = vit
        for layer in self.vit.encoder.layers:
            layer.register_forward_hook(self._hook)
        self.y_list = []
        self._logits = None

    def _hook(self, m, x, y):
        self._logits = y[:, 0, :]
        y = y[:, 1:, :]  # (n, 169, 768) (n, 168+1, 768)  (n, h*w+1, d)
        s = int(math.sqrt(y.shape[1]))
        y = rearrange(y, 'n (h w) d -> n d h w', h=s, w=s)
        self.y_list.append(y)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        self.y_list = []
        self.vit(x)
        return self._logits, self.y_list


class SwinAdapter(nn.Module):

    def __init__(self, backbone: nn.Module):
        super(SwinAdapter, self).__init__()

        self.feature_layers = backbone.features
        self.ch_out_list = []
        for output in self(torch.rand((1, 3, 224, 224), dtype=torch.float32)):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            _, c, h, w = output.shape
            self.ch_out_list.append(int(c))

    def forward(self, x: torch.Tensor):
        y_list = []
        y = x
        for idx, layer in enumerate(self.feature_layers.children()):
            y = layer(y)
            y_list.append(rearrange(y, 'n h w d -> n d h w'))
        return y_list
