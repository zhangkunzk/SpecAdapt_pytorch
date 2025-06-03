#!/usr/bin/env python3

from typing import Optional

import numpy as np
import torch
from torch import nn

from .common import is_target_group, FlatLikeSquare, AbstractDecomposition, grad_norm_
from ..adamw import AdamW

__all__ = [
    'XAdamW'
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


class Decomposition(AbstractDecomposition):

    def __init__(self, original_p: torch.Tensor, r, mh, mw, drop_rate=0):
        super().__init__(original_p, r, FlatLikeSquare(original_p.shape))
        self.h, self.w = self.flat.target_shape[-2:]
        self.mh = mh
        self.mw = mw
        self.dp = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def _decompose(self):
        with torch.no_grad():
            rh = rw = self.r
            if isinstance(self.r, float):
                rh = int(self.h * self.r)
                rw = int(self.w * self.r)
            idx_h = torch.randperm(self.h)[:rh].to(self.p.device)
            idx_w = torch.randperm(self.w)[:rw].to(self.p.device)

            mh = self.mh[:, :self.h]
            mw = self.mw[:, :self.w]
            freq = mh @ self.p @ mw.T

            mh = self.mh[idx_h, :self.h]
            mw = self.mw[idx_w, :self.w]
            freq = freq.index_select(-2, idx_h).index_select(-1, idx_w)
            p0 = self.p - mh.T @ freq @ mw

            self.p0 = p0
            self.idx_h = idx_h
            self.idx_w = idx_w
            self.freq = freq

            self.params = [freq]

    def _compose(self):
        with torch.enable_grad():
            mh = self.mh[self.idx_h, :self.h]
            mw = self.mw[self.idx_w, :self.w]
            self.p = self.p0 + self.dp(mh.T @ self.freq @ mw)

    def propagate_grad(self):
        super().propagate_grad()
        with torch.no_grad():
            grad_norm_(self.freq, (0, 1))


class XAdamW(AdamW):

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

        dtype, device = None, None
        max_h, max_w = 0, 0
        for group in self.param_groups:
            if not is_target_group(group):
                continue
            for p in group['params']:
                if len(p.shape) < 2:
                    continue
                h, w = FlatLikeSquare(p.shape).target_shape[-2:]
                if h > max_h:
                    max_h = h
                if w > max_w:
                    max_w = w
                if dtype is None or device is None:
                    dtype = p.dtype
                    device = p.device

        if max_h > 0 and max_w > 0:
            # mh = create_dct_matrix(max_h, dtype=dtype, device=device)
            # mw = create_dct_matrix(max_w, dtype=dtype, device=device)
            # mh.requires_grad = True
            # mw.requires_grad = True
            m = create_dct_matrix(max(max_h, max_w), dtype=dtype, device=device)
            m.requires_grad = True

            self.decompositions = []
            for group in self.param_groups:
                if not is_target_group(group):
                    continue
                params = group['params']
                group['params'] = []
                for p in params:
                    if len(p.shape) < 2:
                        continue
                    decomposition = Decomposition(p, r, m, m)
                    self.decompositions.append(decomposition)
                    decomposition.init()
                    group['params'].extend(decomposition.params)

            self.param_groups[-1]['params'] += [m]

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


def test():
    a = torch.stack(torch.meshgrid([torch.arange(3), torch.arange(4)], indexing='ij'), -1)
    idx_h = torch.as_tensor([2, 0, 0])
    idx_w = torch.as_tensor([2, 1, 1])
    b = a[idx_h, ...][:, idx_w]
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            print(b[i, j], end='\t')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(test())
