#!/usr/bin/env python3


from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import init

from .common import is_target_group, FlatLikeSquare, AbstractDecomposition, grad_norm_
from ..adamw import AdamW

__all__ = [
    'SAdamW'
]


class DecompositionContext(object):

    def __init__(self, r, weight_dropout=0.1):
        self.r = r
        self.weight_dropout = weight_dropout
        self.decomposition_dict = {}
        self.sh = None

    def add(self, p):
        if self.sh is None:
            self.sh = torch.empty((self.r, self.r * 72), dtype=p.dtype, device=p.device)
            init.orthogonal_(self.sh)
        self.decomposition_dict[id(p)] = SVDDecomposition(p, self.r, self.weight_dropout, self.sh)

    def get(self, p):
        return self.decomposition_dict[id(p)]

    def decompositions(self):
        return self.decomposition_dict.values()

    def init(self):
        max_ent = max(self.decomposition_dict.values(), key=lambda _x: _x.ent).ent
        min_ent = min(self.decomposition_dict.values(), key=lambda _x: _x.ent).ent
        for decomposition in self.decomposition_dict.values():
            decomposition.ent = (decomposition.ent - min_ent) / (max_ent - min_ent)
            decomposition.init()

    def propagate_grad(self):
        for decomposition in self.decomposition_dict.values():
            decomposition.propagate_grad()

    def update(self):
        for decomposition in self.decomposition_dict.values():
            decomposition.update()

    def zero_grad(self, set_to_none):
        for decomposition in self.decomposition_dict.values():
            decomposition.zero_grads(set_to_none=set_to_none)


class SVDDecomposition(AbstractDecomposition):

    def __init__(self, p, r, weight_dropout, sh=None):
        super().__init__(p, r, FlatLikeSquare(p.shape))
        self.weight_dropout = weight_dropout
        self.sh = sh
        with torch.no_grad():
            self.u, self.s, self.vh = torch.linalg.svd(self.p)
            p = self.s / (self.s.sum() + 1e-10)
            self.ent = -(p * (p + 1e-10).log()).sum()

    def _decompose(self):
        with torch.no_grad():
            d = int(self.s.shape[-1])
            max_n, min_n = 0.8, 0.01
            n = int(d * ((max_n - min_n) * self.ent + min_n))
            u1 = self.u[:, :n]
            s1 = self.s[:n]
            vh1 = self.vh[:n, :]
            a = torch.zeros((n, self.r), dtype=self.p.dtype, device=self.p.device)
            b = torch.zeros((self.r, n), dtype=self.p.dtype, device=self.p.device)
            init.kaiming_uniform_(b, float(np.sqrt(5)))

            self.p0 = self.p.clone()
            self.u1 = u1
            self.s1 = s1
            self.vh1 = vh1
            self.a = a
            self.b = b

            self.params = [self.a, self.b]
            if self.sh is not None:
                self.params.append(self.sh)

    def _compose(self):
        with torch.enable_grad():
            u = self.u1 @ self.a
            vh = self.b @ self.vh1
            if self.sh is not None:
                dp = u @ (self.sh @ self.sh.T) @ vh
            else:
                dp = u @ vh
            if self.weight_dropout > 0.0:
                dp = F.dropout(dp, self.weight_dropout)
            self.p = self.p0 + dp

    def propagate_grad(self):
        super().propagate_grad()
        with torch.no_grad():
            grad_norm_(self.a, 0)
            grad_norm_(self.b, 1)


class SAdamW(AdamW):

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

        self.context = DecompositionContext(r)
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            for p in group['params']:
                if len(p.shape) < 2:
                    continue
                self.context.add(p)
        self.context.init()

        unique = set()
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            params = group['params']
            group['params'] = []
            for p in params:
                if len(p.shape) < 2:
                    continue
                decomposition = self.context.get(p)
                for _p in decomposition.params:
                    if id(_p) in unique:
                        continue
                    group['params'].append(_p)
                    unique.add(id(_p))

    def step(self, closure=None):
        self.context.propagate_grad()
        super().step(closure)
        self.context.update()

    def zero_grad(self, set_to_none=False) -> None:
        self.context.zero_grad(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
