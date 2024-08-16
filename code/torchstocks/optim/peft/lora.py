#!/usr/bin/env python3

from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import init

from .common import is_target_group, AbstractDecomposition, FlatLikeSquare, FlatIdentity, layer_grad_norm_
from ..adamw import AdamW

__all__ = [
    'LoRAAdamW'
]


class Decomposition1d(AbstractDecomposition):

    def __init__(self, p):
        super().__init__(p, FlatIdentity())

    def _decompose(self):
        with torch.no_grad():
            self.p0 = torch.clone(self.original_p)
            self.scale = torch.ones((), dtype=self.p.dtype, device=self.p.device)
            self.bias = torch.zeros((), dtype=self.p.dtype, device=self.p.device)

            self.params = [self.scale, self.bias]

    def _compose(self):
        with torch.enable_grad():
            self.p = self.p0 * self.scale + self.bias


class DecompositionNd(AbstractDecomposition):

    def __init__(self, p, r, weight_dropout=0.1, context=None):
        super().__init__(p, FlatLikeSquare(p.shape))
        self.r = r
        self.wd = weight_dropout
        self.context = context

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
            if self.training and self.wd > 0.0:
                d = F.dropout(d, self.wd)
            self.p = self.p0 + d

    def pre_step(self):
        super().pre_step()
        with torch.no_grad():
            layer_grad_norm_(self.u, 0)
            layer_grad_norm_(self.v, 1)


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
                if len(p.shape) == 0:
                    continue
                if len(p.shape) == 1:
                    decomposition = Decomposition1d(p)
                else:
                    decomposition = DecompositionNd(p, r)
                self.decompositions.append(decomposition)
                decomposition.init()
                for _p in decomposition.params:
                    group['params'].append(_p)

    def step(self, closure=None):
        for decomposition in self.decompositions:
            decomposition.pre_step()
        super().step(closure)
        for decomposition in self.decompositions:
            decomposition.post_step()

    def zero_grad(self, set_to_none=False) -> None:
        for decomposition in self.decompositions:
            decomposition.zero_grads(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)

    def train(self, training=True):
        for decomposition in self.decompositions:
            decomposition.train(training)

    def eval(self):
        for decomposition in self.decompositions:
            decomposition.eval()
