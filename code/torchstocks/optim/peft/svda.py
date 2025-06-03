#!/usr/bin/env python3

import math
from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import init

from .common import is_target_group, FlatLikeSquare, AbstractDecomposition, grad_norm_, FlatIdentity
from ..adamw import AdamW

__all__ = [
    'SAdamW'
]


def l1_decay_(x, decay):
    xs = x.sign()
    x.sub_(xs * decay)
    x[x.sign() != xs] = 0
    return x


def l2_decay_(x, decay):
    return x.mul_(1 - decay)


class WrapperNd(AbstractDecomposition):

    def __init__(
            self,
            p,
            rank,
            max_rank=512,
            extra_decay=0.2,
            weight_dropout=0.1,
            sh=None
    ) -> None:
        super().__init__(p, FlatLikeSquare(p.shape))
        self.rank = rank
        self.max_rank = max_rank
        self.extra_decay = extra_decay
        self.weight_dropout = weight_dropout
        self.sh = sh

        self.lr = None
        self.weight_decay = None

    def _decompose(self):
        with torch.no_grad():
            max_rank = min(*self.flat.target_shape, self.max_rank)
            self.u, self.s, v = torch.svd_lowrank(self.p, q=max_rank)
            self.vh = v.mT
            d = int(self.s.shape[0])

            u1 = self.u[:, :d]
            s1 = self.s[:d]
            vh1 = self.vh[:d, :]
            a = torch.zeros((d, self.rank), dtype=self.p.dtype, device=self.p.device)
            b = torch.zeros((self.rank, d), dtype=self.p.dtype, device=self.p.device)
            std = math.sqrt(1.0 / d)
            init.trunc_normal_(b, 0, std, -2 * std, 2 * std)

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
                dp = u @ self.sh @ vh
            else:
                dp = u @ vh
            if self.training and self.weight_dropout > 0.0:
                dp = F.dropout(dp, self.weight_dropout)
            self.p = self.p0 + dp

    def pre_step(self):
        super().pre_step()
        with torch.no_grad():
            grad_norm_(self.a, 0)
            grad_norm_(self.b, 1)

            singular_decay = 1 - self.s1 / self.s1.max()
            weight_decay = self.extra_decay * self.weight_decay
            l1_decay_(self.a, self.lr * weight_decay * singular_decay[:, None])
            l1_decay_(self.b, self.lr * weight_decay * singular_decay)
            # l2_decay_(self.a, self.lr * weight_decay * singular_decay[:, None])
            # l2_decay_(self.b, self.lr * weight_decay * singular_decay)


class Wrapper1d(AbstractDecomposition):

    def __init__(
            self,
            p,
            extra_decay=1,
            weight_dropout=0.1
    ) -> None:
        super().__init__(p, FlatIdentity())
        self.extra_decay = extra_decay
        self.weight_dropout = weight_dropout

        self.lr = None
        self.weight_decay = None

    def _decompose(self):
        with torch.no_grad():
            self.p0 = self.original_p.clone()
            self.p1 = torch.zeros_like(self.p0)

            self.params = [self.p1]

    def _compose(self):
        with torch.enable_grad():
            dp = self.p1
            if self.training and self.weight_dropout > 0.0:
                dp = F.dropout(dp, self.weight_dropout)
            self.p = self.p0 + dp

    def pre_step(self):
        super().pre_step()
        with torch.no_grad():
            weight_decay = self.extra_decay * self.weight_decay
            l1_decay_(self.p1, self.lr * weight_decay)
            # l2_decay_(self.p1, self.lr * weight_decay)


class Wrapper1dScaleShift(AbstractDecomposition):

    def __init__(self, p):
        super().__init__(p, FlatIdentity())

    def _decompose(self):
        with torch.no_grad():
            self.p0 = self.original_p.clone()
            self.scale = torch.zeros((), dtype=self.p.dtype, device=self.p.device)
            self.shift = torch.zeros((), dtype=self.p.dtype, device=self.p.device)

            self.params = [self.scale, self.shift]

    def _compose(self):
        with torch.enable_grad():
            self.p = self.p0 * (1 + self.scale) + self.shift


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

        sh = None
        wrappers_1d = {}
        wrappers_nd = {}
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            for p in group['params']:
                if len(p.shape) < 2:
                    wrappers_1d[id(p)] = Wrapper1d(p)
                else:
                    if sh is None:
                        sh = torch.eye(r, dtype=p.dtype, device=p.device)
                        # init.orthogonal_(sh)
                    wrappers_nd[id(p)] = WrapperNd(p, r, sh=sh)

        for wrapper in wrappers_1d.values():
            wrapper.init()
        for wrapper in wrappers_nd.values():
            wrapper.init()

        wrappers = {**wrappers_1d, **wrappers_nd}
        unique = set()
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            params = group['params']
            group['params'] = []
            group['wrappers'] = {}
            for p in params:
                wrapper = wrappers.get(id(p))
                if wrappers is None:
                    continue
                group['wrappers'][id(p)] = wrapper
                for _p in wrapper.params:
                    if id(_p) in unique:
                        continue
                    unique.add(id(_p))
                    group['params'].append(_p)

        self.r = r
        self.sh: torch.Tensor = sh

    def step(self, closure=None):
        for group in self.param_groups:
            for wrapper in group.get('wrappers', {}).values():
                wrapper.lr = group['lr']
                wrapper.weight_decay = group['weight_decay']
                wrapper.pre_step()
        super().step(closure)
        for group in self.param_groups:
            for wrapper in group.get('wrappers', {}).values():
                wrapper.post_step()

    def zero_grad(self, set_to_none=False) -> None:
        for group in self.param_groups:
            for wrapper in group.get('wrappers', {}).values():
                wrapper.zero_grads(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)

    def train(self, training=True):
        for group in self.param_groups:
            for wrapper in group.get('wrappers', {}).values():
                wrapper.train(training)

    def eval(self):
        for group in self.param_groups:
            for wrapper in group.get('wrappers', {}).values():
                wrapper.eval()
