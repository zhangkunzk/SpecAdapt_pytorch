#!/usr/bin/env python3

"""
@author: xi
@since: 2022-03-24
"""

from typing import Sequence, Tuple, Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from torchstocks import nn as nn_
from torchstocks.models import imagenet as backbone_module
from torchstocks.models.adapters import ResnetAdapter
from torchstocks.nn import memory as memory_module
from torchstocks.nn.attention import MultiHeadAttention2d
from torchstocks.nn.memory import MemoryBank
from torchstocks.nn.vision import PFEDecoder


class Model(nn.Module):

    def __init__(
            self,
            backbone: str = 'resnet34',
            memory: str = 'SoftKMeansMemory',
            feat_layers: Sequence[int] = (3, 2, 1),
            prior_layers: Sequence[int] = (4,),
            mem_size: int = 128,  # The number of prototypes in the memory.
            num_heads: int = 8,
            head_size: int = 64,
            decoder_size: int = 256,
            decoder_depth: int = 4,
            threshold: float = 0.5,
            dropout: float = 0.0,
            criterion: Literal['CrossEntropyLoss', 'FocalLoss'] = 'FocalLoss',
            aux_weight: float = 1.0
    ) -> None:
        super(Model, self).__init__()
        self.network = Network(
            backbone=backbone,
            memory=memory,
            feat_layers=feat_layers,
            prior_layers=prior_layers,
            mem_size=mem_size,
            num_heads=num_heads,
            head_size=head_size,
            decoder_size=decoder_size,
            decoder_depth=decoder_depth,
            threshold=threshold,
            dropout=dropout
        )
        if criterion == 'CrossEntropyLoss':
            self.criterion = CrossEntropyLoss(aux_weight)
        elif criterion == 'FocalLoss':
            self.criterion = FocalLoss(aux_weight)
        else:
            raise ValueError(f'Invalid Loss "{criterion}".')

    def forward(self, inputs, targets=None):
        if isinstance(inputs, tuple):
            task, sx, sy, qx = inputs
        elif isinstance(inputs, dict):
            task = inputs['task']
            sx = inputs['sx']
            sy = inputs['sy']
            qx = inputs['qx']
        else:
            raise ValueError(f'Unsupported inputs type. Expect tuple or dict, got {type(inputs)}')

        output, aux_list = self.network(task=task, qx=qx, sx=sx, sy=sy)

        if targets is None:
            return output
        else:
            if isinstance(targets, tuple):
                qy, = targets
            elif isinstance(targets, dict):
                qy = targets['qy']
            elif isinstance(targets, torch.Tensor):
                qy = targets
            else:
                raise ValueError(f'Unsupported targets type. Expect tuple, dict or Tensor, got {type(inputs)}')

            loss = self.criterion(output, qy, aux_list)
            return loss

    def reset_memories(self):
        self.network.reset_memories()


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels):
        super(GroupNorm, self).__init__(32, num_channels)


class Network(nn.Module):

    def __init__(
            self,
            backbone: str = 'resnet34',
            memory: str = 'SoftKMeansMemory',
            feat_layers: Sequence[int] = (3, 2, 1),
            prior_layers: Sequence[int] = (4,),
            mem_size: int = 128,
            num_heads: int = 8,
            head_size: int = 64,
            decoder_size: int = 256,
            decoder_depth: int = 4,
            threshold: float = 0.5,
            dropout: float = 0.0
    ) -> None:
        super(Network, self).__init__()
        self.backbone_name = backbone
        self.memory_name = memory
        self.feat_layers = feat_layers
        self.prior_layers = prior_layers
        self.mem_size = mem_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.decoder_size = decoder_size
        self.decoder_depth = decoder_depth
        self.threshold = threshold
        self.dropout = dropout

        self.backbone = getattr(backbone_module, self.backbone_name)(pretrained=True)
        if isinstance(self.backbone, backbone_module.ResNet):
            self.backbone = ResnetAdapter(self.backbone)
        else:
            raise RuntimeError(f'Unsupported backbone {self.backbone}.')

        self.backbone_ch_out_list = self.backbone.ch_out_list
        num_outputs = len(self.backbone_ch_out_list)
        self.feat_layers = [i if i >= 0 else i + num_outputs for i in self.feat_layers]
        self.prior_layers = [i if i >= 0 else i + num_outputs for i in self.prior_layers]
        self.all_layers = list({*self.feat_layers, *self.prior_layers})

        Memory = getattr(memory_module, self.memory_name)
        self.memories = nn.ModuleDict()
        for i in self.all_layers:
            self.memories[str(i)] = MemoryBank(
                Memory,
                mem_size=self.mem_size,
                feat_size=self.backbone_ch_out_list[i],
                num_classes=2,  # "fg" and "bg"
                memory_kwargs={}
            )

        self.attentions = nn.ModuleDict()
        for i in self.feat_layers:
            self.attentions[str(i)] = MultiHeadAttention2d(
                feat_size=self.backbone_ch_out_list[i],
                num_heads=self.num_heads,
                head_size=self.head_size,
                num_bottlenecks=0,
                requires_out=False,
                dropout=self.dropout,
                Norm=GroupNorm
            )

        self.decoder = PFEDecoder(
            ch_in_list=[self.num_heads * self.mem_size * 2] * len(self.feat_layers),
            ch_hid=self.decoder_size,
            ch_out=2,  # "fg" and "bg"
            ch_prior=len(self.prior_layers),
            hw_hid_list=None,
            hw_out=60,
            depth=self.decoder_depth,
            dropout=self.dropout,
            Norm=GroupNorm
        )

    def train(self, mode: bool = True):
        super(Network, self).train(mode)
        self.backbone.train(False)
        return self

    def reset_memories(self):
        for memory in self.memories.values():
            memory.reset()

    def forward(
            self,
            task,
            qx: torch.Tensor,  # (n, c, h, w)
            sx: torch.Tensor,  # (n, k, c, h, w)
            sy: torch.Tensor,  # (n, k, h, w)
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        with torch.no_grad():
            #
            # get features for query
            query_dict = {}
            query_out_list = self.backbone(qx)
            for i in self.all_layers:
                query_dict[i] = query_out_list[i]

            #
            # get features for support
            supp_dict = {}
            if sx is not None and sy is not None:
                flat_sx = rearrange(sx, 'n k c h w -> (n k) c h w')
                supp_out_list = self.backbone(flat_sx)
                flat_sm = rearrange(F.one_hot(sy, 2).float(), 'n k h w c -> (n k) c h w')
                for i in self.all_layers:
                    supp_feat = supp_out_list[i]
                    supp_mask = F.interpolate(flat_sm, supp_feat.shape[2:4], mode='area')
                    supp_feat = rearrange(supp_feat, '(n k) c h w -> n (k h w) c', n=sx.shape[0])
                    supp_mask = rearrange(supp_mask, '(n k) c h w -> n c (k h w)', n=sx.shape[0])
                    for task_name, supp_feat_t, supp_mask_t in zip(task, supp_feat, supp_mask):
                        self.memories[str(i)].write(task_name, [
                            supp_feat_t[supp_mask_t[0] >= self.threshold],
                            supp_feat_t[supp_mask_t[1] >= self.threshold],
                        ])
            for i in self.all_layers:
                supp_dict[i] = torch.stack([
                    self.memories[str(i)].read(task_name)
                    for task_name in task
                ])

            #
            # compute prior map
            prior_list = []
            for i in self.prior_layers:
                supp_feat = supp_dict[i]
                query_feat = query_dict[i]
                prior_list.append(self.make_prior(supp_feat[:, 1, :, :], query_feat))
            prior = torch.concat(prior_list, 1)

        #
        # compute scores
        score_list = []
        for i in self.feat_layers:
            supp_feat = rearrange(supp_dict[i], 'n c m d -> n (c m) d')
            query_feat = query_dict[i]
            score = self.attentions[str(i)](query_feat, supp_feat)[0]
            score = rearrange(score, 'n k (c m) h w -> n k c m h w', m=self.mem_size)
            if self.training:
                perm = torch.randperm(score.shape[3])
                score = score[:, :, :, perm, :, :]
            score = rearrange(score, 'n k c m h w -> n (k c m) h w')
            score_list.append(score)

        y, y_list = self.decoder(score_list, prior)
        return y, y_list

    @staticmethod
    def make_prior(
            supp_feat: torch.Tensor,  # (n, m, d)
            query_feat: torch.Tensor,  # (n, d, h, w)
            eps: float = 1e-6
    ) -> torch.Tensor:
        supp_feat[:, :, :] = supp_feat[:, torch.randperm(supp_feat.shape[1]), :]
        n, c, h, w = query_feat.shape
        supp_feat = F.normalize(supp_feat, 2, 2)
        query_feat = F.normalize(query_feat, 2, 1)
        query_feat = query_feat.view((n, c, -1))  # (n, c, hw)
        sim = torch.einsum('nca,nmc->nam', query_feat, supp_feat)  # (n, hw, m)
        sim = F.adaptive_max_pool1d(sim, 71).mean(2)
        sim_min = sim.min(1, keepdim=True)[0]  # (n, 1)
        sim_max = sim.max(1, keepdim=True)[0]  # (n, 1)
        prior = (sim - sim_min) / (sim_max - sim_min + eps)
        prior = prior.view((n, 1, h, w))  # (n, 1, h, w)
        return prior


class CrossEntropyLoss(nn.Module):

    def __init__(self, aux_weight: float = 1.0):
        super(CrossEntropyLoss, self).__init__()
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            aux_list: Sequence[torch.Tensor] = None
    ) -> torch.Tensor:
        target_size = (target.shape[1], target.shape[2])
        output = F.interpolate(output, target_size, mode='bilinear', align_corners=True)
        loss = self.ce(output, target)

        if aux_list:
            aux_loss = sum([
                self.ce(F.interpolate(aux, target_size, mode='bilinear', align_corners=True), target)
                for aux in aux_list
            ]) / len(aux_list)
            loss = loss + self.aux_weight * aux_loss
        return loss


class FocalLoss(nn.Module):

    def __init__(
            self,
            aux_weight: float = 1.0,
            gamma: int = 1
    ) -> None:
        super(FocalLoss, self).__init__()
        self.aux_weight = aux_weight
        self.focal = nn_.FocalLoss(gamma=gamma)

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            aux_list: Sequence[torch.Tensor] = None
    ) -> torch.Tensor:
        target_size = (target.shape[1], target.shape[2])
        output = F.interpolate(output, target_size, mode='bilinear', align_corners=True)
        loss = self.focal(output, target)

        if aux_list:
            aux_loss = sum([
                self.focal(F.interpolate(aux, target_size, mode='bilinear', align_corners=True), target)
                for aux in aux_list
            ]) / len(aux_list)
            loss = loss + self.aux_weight * aux_loss
        return loss
