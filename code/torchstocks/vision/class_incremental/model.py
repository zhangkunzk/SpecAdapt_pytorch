#!/usr/bin/env python3

"""
@author: Howie
@since: 2022-08-01
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from torchstocks.models import cifar, imagenet
from torchstocks.models.adapters import ResnetAdapter

__all__ = [
    'BaseModel'
]


class GroupNorm(nn.GroupNorm):

    def __init__(self, n):
        super(GroupNorm, self).__init__(16, n)


class BaseModel(nn.Module):

    def __init__(
            self,
            backbone_name: str,
            num_class: int,
            Norm=GroupNorm,
            NonLin=nn.SiLU,
    ) -> None:
        super(BaseModel, self).__init__()
        self.num_class = num_class
        self.Norm = Norm
        self.NonLin = NonLin

        assert hasattr(imagenet, backbone_name)
        fn = getattr(imagenet, backbone_name)
        backbone_obj = fn(pretrained=True)
        # backbone_obj = fn(num_classes=10)
        self.backbone = ResnetAdapter(backbone_obj)
        # self._recursive_replace(self.backbone)
        dummy = torch.rand((3, 3, 32, 32))
        dummy = self.backbone(dummy)[-1]
        self.hide_size = dummy.shape[1]  # * dummy.shape[2] * dummy.shape[3]
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.num_class = num_class
        self.norm = nn.LayerNorm(self.hide_size)
        # self.norm = nn.BatchNorm1d(self.hide_size)
        self.fc = nn.Linear(in_features=self.hide_size, out_features=num_class, bias=False)
        self.sessions = None
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.avgpooling(x)
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
