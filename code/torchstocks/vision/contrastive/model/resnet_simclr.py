#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-2-2
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from torchstocks.models.adapters import ResnetAdapter
from torchstocks.common.contrastive.model.simclr import NTCrossEntropyLoss, ProjectionHead


class ProjectionHeadWithBN(nn.Module):

    def __init__(
            self,
            emb_size: int,
            head_size: int,
            skip_conn: bool = False
    ) -> None:
        super(ProjectionHeadWithBN, self).__init__()
        self.skip_conn = skip_conn
        self.hidden = nn.Linear(emb_size, emb_size, bias=False)
        self.bn = nn.BatchNorm1d(emb_size)
        self.non_lin = nn.ReLU()
        self.out = nn.Linear(emb_size, head_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h0 = h
        h = self.hidden(h)
        if self.skip_conn:
            h = h + h0
        h = self.bn(h)
        h = self.non_lin(h)
        h = self.out(h)
        return h


class Model(nn.Module):

    def __init__(
            self,
            head_size: int,
            network: nn.Module,
            skip_conn: bool = False,
            temperature: float = 0.3
    ) -> None:
        super(Model, self).__init__()
        self.backbone = ResnetAdapter(network)
        emb_size = self.backbone.ch_out_list[-1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = ProjectionHead(emb_size, head_size, skip_conn)
        self.criterion = NTCrossEntropyLoss(temperature)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x2 is None:
            y1 = self.avgpool(self.backbone(x1)[-1])
            return torch.flatten(y1, 1)
        else:
            y1 = self.avgpool(self.backbone(x1)[-1])
            y2 = self.avgpool(self.backbone(x2)[-1])
            y1 = torch.flatten(y1, 1)
            y2 = torch.flatten(y2, 1)
            z1 = self.proj_head(y1)
            z2 = self.proj_head(y2)
            return self.criterion(z1, z2)


class NTCrossEntropyLossV2(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""

    def __init__(
            self,
            temperature: float = 0.3,
            eps: float = 1e-10
    ) -> None:
        super(NTCrossEntropyLossV2, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        return nt_xcent_loss(z1, z2, self.temperature, self.eps)


def nt_xcent_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.3,
        eps: float = 1e-10
) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1
    # compute the similarity matrix
    z = torch.concat((z1, z2), dim=0)
    z = F.normalize(z, dim=1)
    sim_mat = z @ z.T

    sim_mat.fill_diagonal_(float('-inf'))
    scaled_prob_mat = F.softmax(sim_mat / temperature, dim=1)
    positive_1 = torch.diagonal(scaled_prob_mat, offset=batch_size)
    positive_2 = torch.diagonal(scaled_prob_mat, offset=-batch_size)
    positive = torch.cat((positive_1, positive_2), dim=0)
    log_prob = torch.log(positive + eps)
    return -log_prob.mean()
