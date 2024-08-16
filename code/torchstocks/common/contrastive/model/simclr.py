#!/usr/bin/env python3


from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'SimCLRWrapper',
    'ProjectionHead',
    'NTCrossEntropyLoss',
    'nt_cross_entropy_loss'
]


class SimCLRWrapper(nn.Module):

    def __init__(
            self,
            network: nn.Module,
            emb_size: int,
            head_size: int,
            skip_conn: bool = False,
            temperature: float = 0.3
    ) -> None:
        super(SimCLRWrapper, self).__init__()
        self.network = network
        self.proj_head = ProjectionHead(emb_size, head_size, skip_conn)
        self.criterion = NTCrossEntropyLoss(temperature)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x2 is None:
            return self.network(x1)
        else:
            z1 = self.proj_head(self.network(x1))
            z2 = self.proj_head(self.network(x2))
            return self.criterion(z1, z2)


class ProjectionHead(nn.Module):

    def __init__(
            self,
            emb_size: int,
            head_size: int,
            skip_conn: bool = False
    ) -> None:
        super(ProjectionHead, self).__init__()
        self.skip_conn = skip_conn
        self.hidden = nn.Linear(emb_size, emb_size)
        self.non_lin = nn.ReLU()
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        if self.skip_conn:
            h = h + h0
        h = self.non_lin(h)
        h = self.out(h)
        return h


class NTCrossEntropyLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""

    def __init__(
            self,
            temperature: float = 0.3,
            eps: float = 1e-10
    ) -> None:
        super(NTCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        return nt_cross_entropy_loss(z1, z2, self.temperature, self.eps)


def nt_cross_entropy_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.3,
        eps: float = 1e-10
) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    # compute the similarity matrix
    # values in the diagonal elements represent the similarity between the (POS, POS) pairs
    # while the other values are the similarity between the (POS, NEG) pairs
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_mat = z1 @ z2.T
    scaled_prob_mat = F.softmax(sim_mat / temperature, dim=1)

    # construct a cross-entropy loss to maximize the probability of the (POS, POS) pairs
    log_prob = torch.log(scaled_prob_mat + eps)
    return -torch.diagonal(log_prob).mean()
