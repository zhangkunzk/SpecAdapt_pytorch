#!/usr/bin/env python3

"""Low Rank Adaptation Optimizers

The optimizers in this module can be used as usual ones like SGD, AdamW.

`optimizer = LoRAAdamW(params, r=0.05, lr=1e-3, weight_decay=0.3)`

Here, "params" is the parameter groups. If you want a group use LoRA optimization, you must add a tag named "lora" in
the group. If the tag is not specified, the parameters in this group will be optimized directly.

The argument "r" denotes the rank to decompose the parameter matrix.
It can be an integer or a float number. If it is a float number, the "rank" is computed as `int(r * min(h, w))`, where
"h" and "w" means the height and width of the matrix.

If the parameter is a higher order tensor, it will be transposed and reshaped into a batched matrix. A batched matrix
is a tensor with shape (batch_size, num_rows, num_columns).

If the parameter is a 1d vector, it will not be optimized.
"""

from typing import Union, Optional

import numpy as np
import torch

from .adamw import AdamW
from .sgd import SGD

__all__ = [
    'LoRASGD',
    'LoRAAdamW'
]


class LoRADecomposition(object):

    def __init__(self, p: torch.Tensor, r: Union[int, float]):
        self.p = p
        self.r = r

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
        dtype, device = self.p.dtype, self.p.device

        p0_shape = self.inner_shape
        self.p0 = torch.zeros(p0_shape, dtype=dtype, device=device, requires_grad=False)
        self.p0[...] = self._permute_reshape(self.p)

        h, w = self.inner_shape[-2:]
        if isinstance(self.r, float):
            self.r = int(self.r * min(h, w))
        if self.r > min(h, w) or self.r <= 0:
            raise ValueError(f'Invalid r = {self.r} with matrix shaped ({h}, {w}).')

        u_shape = (*self.inner_shape[:-2], h, self.r)
        self.u = torch.zeros(u_shape, dtype=dtype, device=device, requires_grad=True)

        v_shape = (*self.inner_shape[:-2], self.r, w)
        self.v = torch.normal(0, 1e-3, v_shape, dtype=dtype, device=device, requires_grad=True)

        self.params = [self.u, self.v]

        # print(self.p.shape, p0_shape, u_shape, v_shape)

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
        self.u.grad = p_grad @ self.v.mT
        self.v.grad = self.u.mT @ p_grad

    @torch.no_grad()
    def update(self):
        p = (self.u @ self.v).add_(self.p0)
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


class LoRASGD(SGD):

    def __init__(
            self,
            params,
            r,
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
                    decomposition = LoRADecomposition(p, r)
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
                try:
                    decomposition = LoRADecomposition(p, r)
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
