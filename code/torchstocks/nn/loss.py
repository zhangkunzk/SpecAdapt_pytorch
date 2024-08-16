#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'FocalLoss',
    'EntropyLoss'
]


class FocalLoss(nn.Module):
    """Focal loss
    """

    def __init__(
            self,
            gamma: int = 1,
            dim: int = 1,
            reduction: str = 'mean'
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.dim = dim
        self.reduction = reduction

    def forward(
            self,
            output: torch.Tensor,  # (n, c, ...)
            target: torch.Tensor  # (n, ....)
    ) -> torch.Tensor:
        """Forward
        """
        prob = F.softmax(output, self.dim)
        prob = prob.gather(self.dim, target.unsqueeze(self.dim))  # (n, 1, ...)
        loss = (1.0 - prob).pow(self.gamma) * prob.log()
        assert self.reduction in {'none', 'mean', 'sum'}
        if self.reduction == 'mean':
            return loss.mean().neg()
        elif self.reduction == 'sum':
            return loss.sum().neg()
        else:
            return loss.neg()


class EntropyLoss(nn.Module):
    """Entropy loss
    """

    def __init__(self, original_loss, lmd=1.0, eps=1e-30):
        super(EntropyLoss, self).__init__()
        self.original_loss = original_loss
        self.lmd = lmd
        self.eps = eps

    def forward(self, output, target):
        """Forward
        """
        loss = self.original_loss(output, target)
        if self.lmd is not None and self.lmd != 0:
            p = F.softmax(output, 1)
            h = -(p * (p + self.eps).log()).sum(1).mean()
            loss = loss + self.lmd * h
        return loss
