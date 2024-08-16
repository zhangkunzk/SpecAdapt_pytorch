#!/usr/bin/env python3

from typing import Optional

from torch import optim
from torch.nn.utils import clip_grad_norm_

from torchstocks import dist

__all__ = [
    'SGD'
]


class SGD(optim.SGD):

    def __init__(
            self,
            params,
            lr,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            clip_grad_norm: Optional[float] = None,
            clip_grad_type=float('inf')
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_type = float(clip_grad_type)

    def parameters(self):
        for group in self.param_groups:
            if 'params' in group:
                for p in group['params']:
                    yield p

    def step(self, closure=None):
        dist.sync_grad(self.parameters())
        for p in self.parameters():
            if p.grad is None:
                continue
            if p.grad.isnan().any() or p.grad.isinf().any():
                return
        if self.clip_grad_norm is not None:
            clip_grad_norm_(
                self.parameters(),
                max_norm=self.clip_grad_norm,
                norm_type=self.clip_grad_type
            )
        return super().step(closure)
