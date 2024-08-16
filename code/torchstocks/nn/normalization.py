#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import torch
from torch import nn

from torch.nn import init

__all__ = [
    'LayerNorm',
    'LayerNorm1d',
    'LayerNorm2d',
    'AdaptiveGroupNorm'
]


class LayerNorm(nn.LayerNorm):

    def __init__(
            self,
            num_features,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            normalized_shape=num_features,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype
        )


def layer_norm(x, weight, bias, eps):
    mu = x.mean(1, keepdim=True)
    diff = x - mu
    var = diff.square().mean(1, keepdim=True)
    x = diff / torch.sqrt(var + eps)
    if weight is not None:
        x = weight * x
    if bias is not None:
        x = x + bias
    return x


class LayerNorm1d(nn.Module):

    def __init__(
            self,
            num_features,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(
            x,
            weight=self.weight[:, None] if self.elementwise_affine else None,
            bias=self.bias[:, None] if self.elementwise_affine else None,
            eps=self.eps
        )


class LayerNorm2d(nn.Module):

    def __init__(
            self,
            num_features,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(
            x,
            weight=self.weight[:, None, None] if self.elementwise_affine else None,
            bias=self.bias[:, None, None] if self.elementwise_affine else None,
            eps=self.eps
        )


class AdaptiveGroupNorm(nn.GroupNorm):
    """Adaptive group normalization
    """

    def __init__(
            self,
            num_channels: int,
            num_groups: int = 1,
            eps: float = 1e-5,
            affine: bool = True,
            device=None,
            dtype=None
    ) -> None:
        small_group_size = num_channels // num_groups
        big_group_size = small_group_size + 1
        self.num_groups_big = num_channels % num_groups
        self.num_channels_big = big_group_size * self.num_groups_big
        self.num_groups = num_groups - self.num_groups_big
        self.num_channels = num_channels - self.num_channels_big

        super(AdaptiveGroupNorm, self).__init__(
            num_groups=self.num_groups,
            num_channels=self.num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        )

        self.norm_big = nn.GroupNorm(
            self.num_groups_big,
            self.num_channels_big,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype
        ) if self.num_groups_big != 0 else None

    def forward(self, x: torch.Tensor):
        if self.norm_big is None:
            return super(AdaptiveGroupNorm, self).forward(x)
        else:
            x1 = x[:, :self.num_channels, ...]
            x2 = x[:, self.num_channels:, ...]
            return torch.cat([
                super(AdaptiveGroupNorm, self).forward(x1),
                self.norm_big(x2)
            ], 1)
