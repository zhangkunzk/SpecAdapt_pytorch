#!/usr/bin/env python3


from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import init

from .common import is_target_group, FlatLikeSquare, AbstractDecomposition, grad_norm_
from ..adamw import AdamW


# rank share
__all__ = [
    'RSAdamW'
]


class SVDDecomposition(AbstractDecomposition):

    def __init__(self, p, r, weight_dropout=0.1, context=None, update_rank_rate=0.5):
        super().__init__(p, FlatLikeSquare(p.shape))
        self.r = r
        self.weight_dropout = nn.Dropout(weight_dropout) if weight_dropout > 0 else nn.Identity()
        self.context = context
        self.update_rank_rate = update_rank_rate

    def _decompose(self):
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(self.p)

            d = int(s.shape[-1])
            r = self.r
            if isinstance(r, float):
                r = int(r * d)

            l = int(d * self.update_rank_rate)
            u1 = u[:, -l:]
            s1 = s[-l:]
            vh1 = vh[-l:, :]
            a = torch.zeros((l, r), dtype=self.p.dtype, device=self.p.device)
            b = torch.zeros((r, l), dtype=self.p.dtype, device=self.p.device)
            init.kaiming_uniform_(b, float(np.sqrt(5)))

            self.sh = None
            if self.context is not None:
                if 'sh' not in self.context:
                    self.context['sh'] = torch.empty((r, r), dtype=self.p.dtype, device=self.p.device)
                    init.orthogonal_(self.context['sh'])
                self.sh = self.context['sh']

            self.p0 = self.p.clone()
            self.u1 = u1
            self.s1 = s1
            self.vh1 = vh1
            self.a = a
            self.b = b

            self.params = [self.a, self.b, self.sh]

    def _compose(self):
        with torch.enable_grad():
            dp = self.u1 @ self.a
            if self.sh is not None:
                dp = dp @ (self.sh @ self.sh.T)
            dp = dp @ (self.b @ self.vh1)
            self.p = self.p0 + self.weight_dropout(dp)

    def pre_step(self):
        super().pre_step()
        with torch.no_grad():
            grad_norm_(self.a, 0)
            grad_norm_(self.b, 1)


class RSAdamW(AdamW):

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
            clip_grad_type=float('inf'),
            update_rank_rate=0.5
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_grad_norm=clip_grad_norm,
            clip_grad_type=clip_grad_type,
        )

        self.context = {}
        self.decompositions = []
        unique = set()
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            params = group['params']
            group['params'] = []
            for p in params:
                if len(p.shape) < 2:
                    continue
                else:
                    decomposition = SVDDecomposition(p, r, context=self.context, update_rank_rate=update_rank_rate)
                self.decompositions.append(decomposition)
                decomposition.init()
                for _p in decomposition.params:
                    if id(_p) in unique:
                        continue
                    group['params'].append(_p)
                    unique.add(id(_p))

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
