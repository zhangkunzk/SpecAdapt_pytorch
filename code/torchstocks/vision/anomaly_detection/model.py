#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-09-27
"""

from typing import Tuple, Sequence, Union, Optional

import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F, init
from torchvision import models

from torchstocks.models.adapters import ResnetAdapter
from torchstocks.nn.memory import EuclideanDistance, HeadTailMemory


class Model(nn.Module):

    def __init__(
            self,
            backbone: str = 'wide_resnet50_2',
            indices: Sequence[int] = (1, 2, 3),
            aggr_kernel: int = 3,
            patch_size: Union[int, Tuple[int, int], None] = None,
            mem_size: int = 8192,
            feat_size: Optional[int] = None,
            w_pe=0.0,
            num_nearest: int = 3,
            num_farthest: int = 7,
            mem_args=None,
            use_amp=True
    ) -> None:
        super(Model, self).__init__()
        self.backbone = backbone
        self.indices = indices
        self.aggr_kernel = aggr_kernel
        self.patch_size = patch_size
        self.mem_size = mem_size
        self.feat_size = feat_size
        self.w_pe = w_pe
        self.num_nearest = num_nearest
        self.num_farthest = num_farthest
        self.use_amp = use_amp

        # create backbone and adapter
        if self.backbone.find('resnet') < 0:
            raise RuntimeError('Only ResNets are supported by backbone.')
        try:
            backbone_class = getattr(models, self.backbone)
        except AttributeError:
            raise RuntimeError(f'Cannot find backbone {self.backbone}.')
        self.backbone = ResnetAdapter(backbone_class(pretrained=True))
        self.backbone_depth = max(self.indices) + 1

        # create feature aggregator
        self.aggregator = FeatureAggregator2d(
            ch_out_list=self.backbone.ch_out_list,
            idx_list=self.indices,
            aggr_kernel_list=(self.aggr_kernel, self.aggr_kernel + 2),
            patch_size=self.patch_size,
            proj_size=self.feat_size,
            w_pe=self.w_pe
        )
        self.feat_size = self.aggregator.ch_out

        self.dist_fn = EuclideanDistance(inplace=True)
        if mem_args is None:
            mem_args = {}
        self.memories = MemoryDict(
            HeadTailMemory,
            self.mem_size,
            self.feat_size,
            dist_fn=self.dist_fn,
            **mem_args
        )

    def to(self, *args, **kwargs):
        ret = super(Model, self).to(*args, **kwargs)
        self.memories.kwargs['device'] = next(self.backbone.parameters()).device
        return ret

    def forward(self, x: torch.Tensor, task: Union[str, int] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        task = str(task)
        with torch.no_grad():
            if self.use_amp:
                with amp.autocast():
                    aggr_feat = self.aggregator(self.backbone(x, self.backbone_depth)).float()
            else:
                aggr_feat = self.aggregator(self.backbone(x, self.backbone_depth)).float()

            n, d, h, w = aggr_feat.shape
            flat_feat = aggr_feat.permute((0, 2, 3, 1)).reshape((-1, d))

            dist = self.dist_fn(flat_feat, self.memories[task].read().detach())
            dist = torch.topk(dist, self.num_nearest, 1, largest=False)[0]
            dist = dist.exp().sum(-1)

            dist = dist.reshape((n, -1))
            # dist = dist - dist.min(1, keepdim=True)[0]

            heatmap = dist.square().reshape((n, 1, h, w))
            heatmap = F.interpolate(heatmap, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

            top_dist = torch.topk(dist, self.num_farthest, 1)[0]  # (n, ?)
            ab_score = top_dist[:, 0]
            correction = (1.0 - (top_dist / top_dist.sum(1, keepdim=True))[:, 0])
            ab_score = correction * ab_score  # (n,)
            return heatmap, ab_score

    def update_memory(self, x: torch.Tensor, task: Union[str, int] = 0, lr: float = None):
        task = str(task)
        with torch.no_grad():
            if self.use_amp:
                with amp.autocast():
                    aggr_feat = self.aggregator(self.backbone(x, self.backbone_depth)).float()
            else:
                aggr_feat = self.aggregator(self.backbone(x, self.backbone_depth)).float()
            flat_feat = aggr_feat.permute((0, 2, 3, 1)).reshape((-1, aggr_feat.shape[1]))
            self.memories[task].lr = lr
            return self.memories[task].write(flat_feat)


class MemoryDict(nn.ModuleDict):

    def __init__(self, MemoryType, *args, **kwargs):
        super(MemoryDict, self).__init__()
        self.MemoryType = MemoryType
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, task):
        if task not in self:
            self[task] = self.MemoryType(*self.args, **self.kwargs)
        return super(MemoryDict, self).__getitem__(task)


class FeatureAggregator2d(nn.Module):

    def __init__(
            self,
            ch_out_list: Sequence[int],
            idx_list: Sequence[int],
            aggr_kernel_list: Sequence[int],
            patch_size: Union[int, Tuple[int, int]] = None,
            proj_size: int = None,
            w_pe=0.0
    ) -> None:
        super(FeatureAggregator2d, self).__init__()
        self.ch_out_list = ch_out_list
        self.idx_list = idx_list
        self.aggr_kernel_list = aggr_kernel_list
        self.patch_size = patch_size
        self.proj_size = proj_size
        self.w_pe = w_pe

        assert len(self.ch_out_list) >= len(self.idx_list)
        assert len(self.idx_list) > 0
        assert len(aggr_kernel_list) >= 1

        if self.patch_size is not None:
            if isinstance(self.patch_size, int):
                self.patch_size = (self.patch_size, self.patch_size)
            assert len(self.patch_size) == 2

        concat_size = sum(self.ch_out_list[i] for i in self.idx_list)
        num_concat = len(self.aggr_kernel_list)
        self.ch_out = concat_size * num_concat

        self.rand_proj = None
        if proj_size is not None:
            self.rand_proj = nn.Conv2d(self.ch_out, self.proj_size, (1, 1), (1, 1), (0, 0), bias=False)
            self.rand_proj.weight.requires_grad = False
            init.orthogonal_(self.rand_proj.weight)
            self.ch_out = proj_size

        if w_pe is not None and w_pe > 0:
            self.ch_out += 2

    def forward(self, feat_list: Sequence[torch.Tensor]) -> torch.Tensor:
        y_list = [feat_list[i] for i in self.idx_list]

        size = self.patch_size
        if size is None:
            mid_idx = len(y_list) // 2
            size = (y_list[mid_idx].shape[2], y_list[mid_idx].shape[3])

        z_list = []
        for k in self.aggr_kernel_list:
            for y in y_list:
                y = F.avg_pool2d(y, (k, k), (1, 1), (k // 2, k // 2))
                y = F.interpolate(y, size, mode='bilinear', align_corners=True)
                z_list.append(y)
        z = torch.cat(z_list, 1)

        if self.rand_proj is not None:
            z = self.rand_proj(z)

        # Positional Embedding
        # Append coordinate information to every feature vector.
        # This can improve the performance of "object" tasks when the objects are aligned.
        if self.w_pe is not None and self.w_pe > 0:
            h, w = z.shape[2:]
            pe = torch.stack([
                torch.linspace(0, self.w_pe, h, device=z.device)[:, None].tile((1, w)),
                torch.linspace(0, self.w_pe, w, device=z.device)[None, :].tile((h, 1)),
            ]).tile((z.shape[0], 1, 1, 1))
            z = torch.concatenate([z, pe], 1)
        return z
