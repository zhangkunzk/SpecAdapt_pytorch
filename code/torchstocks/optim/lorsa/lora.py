#!/usr/bin/env python3


from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import init

from .common import is_target_group, AbstractDecomposition, FlatLikeSquare, grad_norm_
from ..adamw import AdamW

__all__ = [
    'LoRAAdamW'
]


class Decomposition(AbstractDecomposition):

    def __init__(self, p, r, drop_rate=0):
        super().__init__(p, r, FlatLikeSquare(p.shape))
        self.dp = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def _decompose(self):
        with torch.no_grad():
            h, w = self.p.shape
            r = self.r
            if isinstance(r, float):
                r = int(r * min(h, w))

            self.p0 = torch.clone(self.p)
            self.u = torch.zeros((h, r), dtype=self.p.dtype, device=self.p.device)
            self.v = torch.zeros((r, w), dtype=self.p.dtype, device=self.p.device)
            init.kaiming_uniform_(self.v, float(np.sqrt(5)))

            self.params = [self.u, self.v]

    def _compose(self):
        with torch.enable_grad():
            d = self.u @ self.v
            self.p = self.p0 + self.dp(d)

    def propagate_grad(self):
        super().propagate_grad()
        with torch.no_grad():
            grad_norm_(self.u, 0)
            grad_norm_(self.v, 1)


class LoRAAdamW(AdamW):

    def __init__(
            self,
            params,
            r,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.3,
            amsgrad=False,
            clip_grad_norm: Optional[float] = None,
            clip_grad_type=float('inf')
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_grad_norm=clip_grad_norm,
            clip_grad_type=clip_grad_type
        )

        self.decompositions = []
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            params = group['params']
            group['params'] = []
            for p in params:
                if len(p.shape) < 2:
                    continue
                decomposition = Decomposition(p, r)
                self.decompositions.append(decomposition)
                decomposition.init()
                group['params'].extend(decomposition.params)

    def step(self, closure=None):
        for decomposition in self.decompositions:
            decomposition.propagate_grad()
        super().step(closure)
        for decomposition in self.decompositions:
            decomposition.update()

    def zero_grad(self, set_to_none=False) -> None:
        for decomposition in self.decompositions:
            decomposition.zero_grads(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
