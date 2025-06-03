#!/usr/bin/env python3


from typing import Union, Optional

import numpy as np
import torch

from .adamw import AdamW
from .sgd import SGD

__all__ = [
    'SSASGD',
    'SSAAdamW'
]


class SSADecompositionV1(object):

    def __init__(self, p: torch.Tensor, r: Union[int, float], largest_singulars=False):
        self.p = p
        self.r = r
        self.largest_singulars = largest_singulars

        with torch.no_grad():
            self._init_permute_reshape()
            self._init_decomposition()

    def _init_permute_reshape(self):
        shape = [int(s) for s in self.p.shape]
        rank = len(shape)
        if rank < 2:
            raise ValueError(f'Expected rank >= 2, got {rank}.')

        dims = sorted([int(i) for i in np.argsort(shape)[-2:]])
        permute = [i for i in range(rank) if i not in dims]
        permute.extend(dims)
        if all(permute[i] == i for i in range(rank)):
            permute = None

        if permute is not None:
            permute_r = [int(i) for i in np.argsort(permute)]
            inner_shape = [shape[i] for i in permute]
        else:
            permute_r = None
            inner_shape = [s for s in shape]

        reshape = None
        reshape_r = None
        if rank > 3:
            reshape = (int(np.prod(inner_shape[:-2])), inner_shape[-2], inner_shape[-1])
            reshape_r = inner_shape
            inner_shape = reshape

        self.permute = permute
        self.reshape = reshape
        self.reshape_r = reshape_r
        self.permute_r = permute_r
        self.inner_shape = inner_shape

    def _init_decomposition(self):
        h, w = self.inner_shape[-2:]
        if isinstance(self.r, float):
            self.r = int(self.r * min(h, w))
        if self.r > min(h, w) or self.r <= 0:
            raise ValueError(f'Invalid r = {self.r} with matrix shaped ({h}, {w}).')

        inner_p = self._permute_reshape(self.p)
        u, s, vh = torch.linalg.svd(inner_p)
        if self.largest_singulars:
            u = u[..., :self.r]
            s = s[..., None, :self.r]
            vh = vh[..., :self.r, :]
        else:
            rank = s.shape[-1]
            u = u[..., rank - self.r:rank]
            s = s[..., None, -self.r:]
            vh = vh[..., rank - self.r:rank, :]
        p0 = inner_p - (u * s) @ vh
        u.requires_grad = True
        vh.requires_grad = True
        s.requires_grad = False
        p0.requires_grad = False

        self.u = u
        self.vh = vh
        self.s = s
        self.st = s.mT
        self.p0 = p0
        self.params = [u, vh]

    def _permute_reshape(self, p):
        if self.permute:
            p = p.permute(self.permute)
        if self.reshape:
            p = p.reshape(self.reshape)
        return p

    def _permute_reshape_r(self, p):
        if self.reshape_r:
            p = p.reshape(self.reshape_r)
        if self.permute_r:
            p = p.permute(self.permute_r)
        return p

    @torch.no_grad()
    def propagate_grad(self):
        p_grad = self._permute_reshape(self.p.grad)
        self.vh.grad = (self.u * self.s).mT @ p_grad
        self.u.grad = p_grad @ (self.st * self.vh).mT

    @torch.no_grad()
    def update(self):
        p = ((self.u * self.s) @ self.vh).add_(self.p0)
        self.p[...] = self._permute_reshape_r(p)

    @torch.no_grad()
    def zero_grads(self, set_to_none=False):
        if set_to_none:
            self.p.grad = None
        else:
            self.p.zero_()


class SSADecomposition(object):

    def __init__(self, p: torch.Tensor, r: Union[int, float], largest_singulars=False):
        self.p = p
        self.r = r
        self.largest_singulars = largest_singulars

        with torch.no_grad():
            self._init_permute_reshape()
            self._init_decomposition()

    def _init_permute_reshape(self):
        shape = [int(s) for s in self.p.shape]
        rank = len(shape)
        if rank < 2:
            raise ValueError(f'Expected rank >= 2, got {rank}.')

        dims = sorted([int(i) for i in np.argsort(shape)[-2:]])
        permute = [i for i in range(rank) if i not in dims]
        permute.extend(dims)
        if all(permute[i] == i for i in range(rank)):
            permute = None

        if permute is not None:
            permute_r = [int(i) for i in np.argsort(permute)]
            inner_shape = [shape[i] for i in permute]
        else:
            permute_r = None
            inner_shape = [s for s in shape]

        reshape = None
        reshape_r = None
        if rank > 3:
            reshape = (int(np.prod(inner_shape[:-2])), inner_shape[-2], inner_shape[-1])
            reshape_r = inner_shape
            inner_shape = reshape

        self.permute = permute
        self.reshape = reshape
        self.reshape_r = reshape_r
        self.permute_r = permute_r
        self.inner_shape = inner_shape

    def _init_decomposition(self):
        h, w = self.inner_shape[-2:]
        if isinstance(self.r, float):
            self.r = int(self.r * min(h, w))
        if self.r > min(h, w) or self.r <= 0:
            raise ValueError(f'Invalid r = {self.r} with matrix shaped ({h}, {w}).')

        inner_p = self._permute_reshape(self.p)
        u0, s, vh0 = torch.linalg.svd(inner_p)
        if self.largest_singulars:
            u0 = u0[..., :self.r]
            s = s[..., None, :self.r]
            vh0 = vh0[..., :self.r, :]
        else:
            rank = s.shape[-1]
            u0 = u0[..., rank - self.r:rank]
            s = s[..., None, -self.r:]
            vh0 = vh0[..., rank - self.r:rank, :]
        p0 = inner_p - (u0 * s) @ vh0
        u0.requires_grad = False
        vh0.requires_grad = False
        s.requires_grad = False
        p0.requires_grad = False

        self.u = torch.zeros_like(u0, requires_grad=True)
        self.vh = torch.zeros_like(vh0, requires_grad=True)
        self.u0 = u0
        self.vh0 = vh0
        self.s = s
        self.st = s.mT
        self.p0 = p0
        self.params = [self.u, self.vh]

    def _permute_reshape(self, p):
        if self.permute:
            p = p.permute(self.permute)
        if self.reshape:
            p = p.reshape(self.reshape)
        return p

    def _permute_reshape_r(self, p):
        if self.reshape_r:
            p = p.reshape(self.reshape_r)
        if self.permute_r:
            p = p.permute(self.permute_r)
        return p

    @torch.no_grad()
    def propagate_grad(self):
        p_grad = self._permute_reshape(self.p.grad)
        self.vh.grad = ((self.u0 + self.u) * self.s).mT @ p_grad
        self.u.grad = p_grad @ (self.st * (self.vh0 + self.vh)).mT

    @torch.no_grad()
    def update(self):
        p = (((self.u0 + self.u) * self.s) @ (self.vh0 + self.vh)).add_(self.p0)
        self.p[...] = self._permute_reshape_r(p)

    @torch.no_grad()
    def zero_grads(self, set_to_none=False):
        if set_to_none:
            self.p.grad = None
        else:
            self.p.zero_()


def is_target_group(group):
    if 'tag' in group:
        tag = group['tag']
        if isinstance(tag, str):
            return tag.lower() == 'low_rank'
        else:
            return 'low_rank' in tag
    elif 'low_rank' in group:
        return group['low_rank']
    else:
        return False


class SSASGD(SGD):

    def __init__(
            self,
            params,
            r,
            largest_singulars=False,
            lr=0.01,
            momentum=0.0,
            weight_decay=1e-4,
            nesterov=False,
            clip_grad_norm: Optional[float] = None,
            clip_grad_type=float('inf')
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
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
                try:
                    decomposition = SSADecomposition(p, r, largest_singulars=largest_singulars)
                except ValueError:
                    continue
                self.decompositions.append(decomposition)
                group['params'].extend(decomposition.params)

    @torch.no_grad()
    def step(self, closure=None):
        for decomposition in self.decompositions:
            decomposition.propagate_grad()
        super().step(closure)
        for decomposition in self.decompositions:
            decomposition.update()

    @torch.no_grad()
    def zero_grad(self, set_to_none=False) -> None:
        for decomposition in self.decompositions:
            decomposition.zero_grads(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)


class SSAAdamW(AdamW):

    def __init__(
            self,
            params,
            r,
            largest_singulars=False,
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
                try:
                    decomposition = SSADecomposition(p, r, largest_singulars=largest_singulars)
                except ValueError:
                    continue
                self.decompositions.append(decomposition)
                group['params'].extend(decomposition.params)

    @torch.no_grad()
    def step(self, closure=None):
        for decomposition in self.decompositions:
            decomposition.propagate_grad()
        super().step(closure)
        for decomposition in self.decompositions:
            decomposition.update()

    @torch.no_grad()
    def zero_grad(self, set_to_none=False) -> None:
        for decomposition in self.decompositions:
            decomposition.zero_grads(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
