#!/usr/bin/env python3

"""
@author: xi
@since: 2021-08-03
"""

import math
from typing import Iterable, Sequence, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .vision import Bottleneck2d, ConvBlock2d

__all__ = [
    'MultiMetricAttention',
    'MultiHeadAttention2d',
    'PyramidMultiHeadAttention2d'
]


class MultiMetricAttention(nn.Module):
    """Multi-metric attention
    """

    def __init__(self,
                 input_size: int,
                 att_size: int,
                 num_att: int,
                 num_groups: int = 32):
        super(MultiMetricAttention, self).__init__()
        self._input_size = input_size
        self._att_size = att_size
        self._num_att = num_att
        self._num_groups = num_groups
        feat_size = att_size * num_att
        num_groups = num_att * self._num_groups
        assert feat_size % num_groups == 0

        self.down = nn.Sequential(
            nn.Conv2d(input_size, feat_size, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups, feat_size),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.res = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups, feat_size),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(feat_size, feat_size, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups, feat_size),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.emb = nn.Conv2d(feat_size, feat_size, kernel_size=(1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, query_feat, proto_feat):
        """Forward
        """
        # query_feat: (n, d, h, w)
        # proto_feat: (n, m, d)
        n, _, h, w = query_feat.shape
        _, m, _ = proto_feat.shape

        query_feat = self.down(query_feat)
        query_feat = self.res(query_feat) + query_feat
        query_feat = self.emb(query_feat)
        query_feat = query_feat.reshape((n, self._num_att, -1, h, w))

        proto_feat = proto_feat.permute((0, 2, 1)).unsqueeze(-1)  # (n, d, m, 1)
        proto_feat = self.down(proto_feat)
        proto_feat = self.res(proto_feat) + proto_feat
        proto_feat = self.emb(proto_feat)
        proto_feat = proto_feat.squeeze(-1).permute((0, 2, 1))
        proto_feat = proto_feat.reshape((n, m, self._num_att, -1))

        query_feat_ = F.normalize(query_feat, 2, 1)
        proto_feat_ = F.normalize(proto_feat, 2, 2)
        feat = torch.einsum('nkdhw,nmkd->nkmhw', query_feat_, proto_feat_)  # (n, k, m, h, w)
        feat = feat / math.sqrt(proto_feat.shape[-1])
        feat = F.softmax(feat, 2)
        feat = self.dropout(feat)
        feat = feat.reshape((n, -1, h, w))
        return feat


class MultiHeadAttention2d(nn.Module):
    """Multi-head attention
    """

    def __init__(
            self,
            feat_size: int,
            num_heads: int = 8,
            head_size: int = 64,
            num_bottlenecks: int = 0,
            requires_out=True,
            proj_out=True,
            Norm=None,
            NonLin=None,
            dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention2d, self).__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_bottlenecks = num_bottlenecks
        self.requires_out = requires_out
        self.proj_out = proj_out
        self.Norm = Norm
        self.NonLin = NonLin
        self.dropout = dropout

        inner_size = self.num_heads * self.head_size
        bottlenecks = [
            Bottleneck2d(
                self.feat_size,
                self.feat_size,
                kernels=(1, 1),
                dropout=self.dropout,
                Norm=self.Norm,
                NonLin=self.NonLin
            )
            for _ in range(self.num_bottlenecks)
        ]
        self.head = nn.Sequential(
            *bottlenecks,
            ConvBlock2d(self.feat_size, inner_size, Norm=Norm, NonLin=NonLin),
            nn.Dropout(dropout)
        )

        if self.requires_out and self.proj_out:
            self.out_project = nn.Conv2d(inner_size, self.feat_size, (1, 1), (1, 1), (0, 0))

    def forward(
            self,
            query: torch.Tensor,  # (n, d, h, w)
            key: torch.Tensor  # (n, m, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (n, k, m, h, w), (n, d, h, w)
        """Forward
        """
        # query feature mapping
        inner_query = self.head(query)  # (n, ks, h, w)
        inner_query = rearrange(inner_query, 'n (k s) h w -> n k s h w', k=self.num_heads)

        # key feature mapping
        key2d = rearrange(key, 'n m d -> n d m 1')
        inner_key = self.head(key2d)  # (n, ks, m, 1)
        inner_key = rearrange(inner_key, 'n (k s) m 1 -> n k s m', k=self.num_heads)

        # compute scores
        inner_query = F.normalize(inner_query, 2, 2)
        inner_key = F.normalize(inner_key, 2, 2)
        score = torch.einsum('n k s h w, n k s m -> n k m h w', inner_query, inner_key)
        score = F.softmax(score, 2)

        out = None
        if self.requires_out:
            out = torch.einsum('n k m h w, n k s m -> n k s h w', score, inner_key)
            out = rearrange(out, 'n k s h w -> n (k s) h w')
            if self.proj_out:
                out = self.out_project(out)

        return score, out


class PyramidMultiHeadAttention2d(nn.Module):
    """Pyramid multi-head attention
    """

    def __init__(
            self,
            feat_sizes: Iterable[int],
            num_heads: int = 8,
            head_size: int = 64,
            num_bottlenecks=0,
            requires_out=True,
            requires_score=False,
            inter_scale_conn=False,
            proj_out=True,
            Norm=None,
            NonLin=None,
            dropout: float = 0.0,
    ) -> None:
        super(PyramidMultiHeadAttention2d, self).__init__()
        self.feat_sizes = list(feat_sizes)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_bottlenecks = num_bottlenecks
        self.requires_out = requires_out
        self.requires_score = requires_score
        self.inter_scale_conn = inter_scale_conn
        self.proj_out = proj_out

        self.heads = nn.ModuleList()
        for feat_size in self.feat_sizes:
            inner_size = self.num_heads * self.head_size
            self.heads.append(nn.Sequential(
                *(Bottleneck2d(feat_size, feat_size, kernels=(1, 1), dropout=dropout, Norm=Norm, NonLin=NonLin)
                  for _ in range(self.num_bottlenecks)),
                ConvBlock2d(feat_size, inner_size, Norm=Norm, NonLin=NonLin),
                nn.Dropout(dropout)
            ))

        self.out_projects = nn.ModuleList()
        if self.requires_out and self.proj_out:
            for feat_size in self.feat_sizes:
                inner_size = self.num_heads * self.head_size
                self.out_projects.append(nn.Conv2d(inner_size, feat_size, (1, 1), (1, 1), (0, 0)))

    def forward(
            self,
            query_list: Sequence[torch.Tensor],  # [(n, d, h, w)]
            key_list: Sequence[torch.Tensor]  # [(n, m, d)]
    ):
        """Forward
        """
        last_inner_query = None
        score_list = []
        out_list = []
        for i, (query, key, head) in enumerate(zip(query_list, key_list, self.heads)):
            k = self.num_heads
            h, w = query.shape[2], query.shape[3]

            # query feature mapping
            inner_query = head(query)  # (n, ks, h, w)
            if self.inter_scale_conn:
                # todo: The feature fusion method is incorrect!
                #       It will change the query space, while the key space stays unchanged.
                #       The unmatched feature spaces cause poor matching score.
                if last_inner_query is not None:
                    last_inner_query = F.interpolate(last_inner_query, (h, w), mode='nearest')  # (n, ks, h, w)
                    inner_query = inner_query + last_inner_query  # (n, ks, h, w)
                last_inner_query = inner_query
            inner_query = rearrange(inner_query, 'n (k s) h w -> n k s h w', k=k)
            # if self.inter_scale_conn and len(score_list) > 0:
            #     inner_query = inner_query + F.interpolate(score_list[-1].mean(2), (h, w), mode='nearest').unsqueeze(2)

            # key feature mapping
            key2d = rearrange(key, 'n m d -> n d m 1')
            inner_key = head(key2d)  # (n, ks, m, 1)
            inner_key = rearrange(inner_key, 'n (k s) m 1 -> n k s m', k=k)

            # compute scores
            inner_query = F.normalize(inner_query, 2, 2)
            inner_key = F.normalize(inner_key, 2, 2)
            score = torch.einsum('n k s h w, n k s m -> n k m h w', inner_query, inner_key)
            score = F.softmax(score, 2)
            score_list.append(score)

            if self.requires_out:
                out = torch.einsum('n k m h w, n k s m -> n k s h w', score, inner_key)
                out = rearrange(out, 'n k s h w -> n (k s) h w')
                if self.proj_out:
                    out = self.out_projects[i](out)
                out_list.append(out)

        if self.requires_out:
            if self.requires_score:
                return out_list, score_list
            else:
                return out_list
        else:
            if self.requires_score:
                return score_list
            else:
                return
