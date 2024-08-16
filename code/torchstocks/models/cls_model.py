#!/usr/bin/env python3

"""
@author: Howie
@since: 2022-08-01
"""

import torch
from torch import nn
from einops import rearrange

from torchstocks.models import cifar
from torchstocks.models.adapters import ResnetAdapter


class GroupNorm(nn.GroupNorm):

    def __init__(self, n):
        super(GroupNorm, self).__init__(16, n)


class ClsModel(nn.Module):

    def __init__(
            self,
            backbone_name: str,
            num_class: int,
            Norm=GroupNorm,
            NonLin=nn.SiLU,
    ) -> None:
        super(ClsModel, self).__init__()
        self.num_class = num_class
        self.Norm = Norm
        self.NonLin = NonLin

        assert hasattr(cifar, backbone_name)
        fn = getattr(cifar, backbone_name)
        backbone_obj = fn(num_classes=num_class)
        self.backbone = ResnetAdapter(backbone_obj)
        # self._recursive_replace(self.backbone)
        dummy = torch.rand((3, 3, 32, 32))
        dummy = self.backbone(dummy)[-1]
        self.hide_size = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]
        self.num_class = num_class
        self.norm = nn.LayerNorm(self.hide_size)
        self.fc = nn.Linear(in_features=self.hide_size, out_features=num_class)
        self.sessions = None

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc(x)
        return x

    def _recursive_replace(self, m: nn.Module):
        for name, child in m.named_children():
            if self.Norm is not None and isinstance(child, nn.BatchNorm2d):
                setattr(m, name, self.Norm(child.num_features))
            if self.NonLin is not None and isinstance(child, nn.ReLU):
                setattr(m, name, self.NonLin())
            self._recursive_replace(child)
