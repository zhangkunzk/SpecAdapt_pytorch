#!/usr/bin/env python3

import numpy as np
import torch

__all__ = [
    'is_target_group',
    'get_slice_by_rank',
    'FlatKeepLastDim',
    'FlatLikeSquare',
    'FlatKeepLongerSide',
    'AbstractDecomposition',
    'grad_norm_'
]


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


def get_slice_by_rank(r, full_rank):
    if isinstance(r, (tuple, list)):
        assert len(r) == 2
        start, stop = r
        if isinstance(start, float):
            start = int(start * full_rank)
        if isinstance(stop, float):
            stop = int(stop * full_rank)
        if isinstance(start, int) and isinstance(stop, int):
            assert 0 <= start < stop <= full_rank
            return start, stop
    else:
        if isinstance(r, float):
            r = int(r * full_rank)
        if isinstance(r, int):
            assert 0 < r <= full_rank
            start = full_rank - r
            stop = full_rank
            return start, stop
    raise ValueError(f'Invalid rank value {r}.')


class FlatKeepLastDim(object):

    def __init__(self, shape):
        shape = tuple(int(s) for s in shape)
        self.source_shape = shape
        self.target_shape = (int(np.prod(shape[:-1])), shape[-1])
        self.need_reshape = self.source_shape != self.target_shape

    def __call__(self, x, inverse=False):
        if inverse:
            if self.need_reshape:
                x = x.reshape(self.source_shape)
        else:
            if self.need_reshape:
                x = x.reshape(self.target_shape)
        return x


class FlatLikeSquare(object):

    def __init__(self, shape, ignore_2d=False):
        shape = tuple(int(s) for s in shape)
        self.source_shape = shape

        if (not ignore_2d) or len(shape) > 2:
            size = int(np.prod(shape))
            factor = int(np.sqrt(size))
            while size % factor != 0:
                factor -= 1
            self.target_shape = (factor, size // factor)
        else:
            self.target_shape = shape

        self.need_reshape = self.source_shape != self.target_shape

    def __call__(self, x, inverse=False):
        if inverse:
            if self.need_reshape:
                x = x.reshape(self.source_shape)
        else:
            if self.need_reshape:
                x = x.reshape(self.target_shape)
        return x


class FlatKeepLongerSide(object):

    def __init__(self, shape):
        shape = [int(s) for s in shape]
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

    def __call__(self, x, inverse=False):
        if inverse:
            if self.reshape_r:
                x = x.reshape(self.reshape_r)
            if self.permute_r:
                x = x.permute(self.permute_r)
        else:
            if self.permute:
                x = x.permute(self.permute)
            if self.reshape:
                x = x.reshape(self.reshape)
        return x


class AbstractDecomposition(object):

    def __init__(self, original_p, r, flatten):
        self.original_p = original_p
        self.r = r
        self.flat = flatten

        self.params = []
        self.p = self.flat(original_p)

    def init(self):
        self._decompose()
        for p in self.params:
            p.requires_grad = True
        self._compose()

    def _decompose(self):
        pass

    def _compose(self):
        pass

    def propagate_grad(self):
        with torch.no_grad():
            flat_g = self.flat(self.original_p.grad)
            self.p.backward(flat_g)

    def update(self):
        p = self._compose()
        if p is None:
            p = self.p
        with torch.no_grad():
            self.original_p[...] = self.flat(p, inverse=True)

    def zero_grads(self, set_to_none=False):
        with torch.no_grad():
            if set_to_none:
                self.original_p.grad = None
            else:
                self.original_p.zero_()


def grad_norm_(p, dim, eps=1e-8):
    g = p.grad
    mu = g.mean(dim, keepdims=True)
    sigma = (g - mu).square_().mean(dim, keepdims=True).sqrt_().add_(eps)
    g.sub_(mu).div_(sigma)
