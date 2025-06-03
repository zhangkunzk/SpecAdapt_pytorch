#!/usr/bin/env python3


from typing import Optional

import torch

from .common import is_target_group, get_slice_by_rank, FlatLikeSquare, AbstractDecomposition
from ..adamw import AdamW

__all__ = [
    'SAdamW'
]


class SVDDecompositionV1(AbstractDecomposition):

    def __init__(self, p, r):
        super().__init__(p, r, FlatLikeSquare(p.shape))

    def _decompose(self):
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(self.p)
            start, stop = get_slice_by_rank(self.r, s.shape[-1])
            u = u[..., start:stop]
            s = s[..., None, start:stop]
            vh = vh[..., start:stop, :]
            p0 = self.p - (u * s) @ vh

            self.p0 = p0
            self.u = u
            self.vh = vh
            self.s = s

            self.params = [u, vh]

    def _compose(self):
        with torch.enable_grad():
            self.p = self.p0 + (self.u * self.s) @ self.vh


class SVDDecomposition(object):

    def __init__(self, p: torch.Tensor, r):
        self.p = p
        self.r = r

        self.flat = FlatLikeSquare(p.shape)
        self._init_decomposition()

    @torch.no_grad()
    def _init_decomposition(self):
        inner_p = self.flat(self.p)
        u0, s, vh0 = torch.linalg.svd(inner_p)
        start, stop = get_slice_by_rank(self.r, s.shape[-1])
        u0 = u0[..., start:stop]
        s = s[..., None, start:stop]
        vh0 = vh0[..., start:stop, :]
        p0 = inner_p - (u0 * s) @ vh0

        self.u = torch.zeros_like(u0, requires_grad=True)
        self.vh = torch.zeros_like(vh0, requires_grad=True)
        self.u0 = u0
        self.vh0 = vh0
        self.s = s
        self.st = s.mT
        self.p0 = p0
        self.params = [self.u, self.vh]

    @torch.no_grad()
    def propagate_grad(self):
        p_grad = self.flat(self.p.grad)
        self.vh.grad = ((self.u0 + self.u) * self.s).mT @ p_grad
        self.u.grad = p_grad @ (self.st * (self.vh0 + self.vh)).mT

    @torch.no_grad()
    def update(self):
        p = (((self.u0 + self.u) * self.s) @ (self.vh0 + self.vh)).add_(self.p0)
        self.p[...] = self.flat(p, inverse=True)

    @torch.no_grad()
    def zero_grads(self, set_to_none=False):
        if set_to_none:
            self.p.grad = None
        else:
            self.p.zero_()


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

        self.decompositions = []
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            params = group['params']
            group['params'] = []
            for p in params:
                if len(p.shape) < 2:
                    continue
                decomposition = SVDDecompositionV1(p, r)
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
