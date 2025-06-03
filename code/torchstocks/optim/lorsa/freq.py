#!/usr/bin/env python3

from typing import Optional

import numpy as np
import torch

from .common import is_target_group, FlatLikeSquare, get_slice_by_rank, AbstractDecomposition
from ..adamw import AdamW

__all__ = [
    'FAdamW'
]


def create_haar_matrix(n, normalized=True):
    n = 2 ** np.ceil(np.log2(n))

    if n <= 2:
        h = np.array([[1, 1], [1, -1]])
    else:
        h = create_haar_matrix(n / 2)
        upper = np.kron(h, [1, 1])
        lower = np.kron(np.eye(len(h)), [1, -1])
        h = np.vstack([upper, lower])
    if normalized:
        h *= np.sqrt(0.5)
    return h


def create_dct_matrix(n, normalize=True, dtype=None, device=None):
    row = np.arange(n)[None, :]
    col = np.arange(n)[:, None]
    m = np.cos(np.pi / n * (row + 0.5) * col)
    if normalize:
        m /= np.linalg.norm(m, 2, 1, keepdims=True)
    m = torch.as_tensor(m, dtype=dtype, device=device)
    return m


class FreqDecomposition(AbstractDecomposition):

    def __init__(self, original_p: torch.Tensor, r):
        super().__init__(original_p, r, FlatLikeSquare(original_p.shape))

    def _decompose(self):
        with torch.no_grad():
            h, w = self.flat.target_shape[-2:]
            dtype, device = self.original_p.dtype, self.original_p.device
            mh = create_dct_matrix(h, dtype=dtype, device=device)[:, :h]
            mw = create_dct_matrix(w, dtype=dtype, device=device)[:, :w]
            freq = mh @ self.p @ mw.T
            start_h, stop_h = get_slice_by_rank(self.r, h)
            start_w, stop_w = get_slice_by_rank(self.r, w)
            mh = mh[start_h:stop_h, :]
            mw = mw[start_w:stop_w, :]
            freq = freq[..., start_h:stop_h, start_w:stop_w]
            p0 = self.p - mh.T @ freq @ mw

            self.p0 = p0
            self.mh = mh
            self.mw = mw
            self.freq = freq

            self.params = [freq]

    def _compose(self):
        self.p = self.p0 + self.mh.T @ self.freq @ self.mw


class FAdamW(AdamW):

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
                decomposition = FreqDecomposition(p, r)
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
