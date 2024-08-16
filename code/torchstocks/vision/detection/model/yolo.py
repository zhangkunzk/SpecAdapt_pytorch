#!/usr/bin/env python3

"""
@author: liying50
@since: 2022-11-03
"""

import torch
from torch import nn
from torchstocks.vision.detection.model.yolov5 import Backbone, Yolo, YoloAnchorFree
from torchstocks.vision.detection.model.yolov5 import YoloLoss, YoloLossAnchorFree


class Model(nn.Module):
    """Define model
    """

    def __init__(
            self,
            image_size: int,
            num_classes: int,
            num_heads: int = 3,
            feat_size: int = 32,
            num_bottlenecks: int = 1,
            inter_mode: str = 'nearest',
            anchor_t: float = 4.0,
            dropout: float = 0.0,
            anchor_free: bool = False,
    ) -> None:
        super(Model, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.feat_size = feat_size
        self.num_bottlenecks = num_bottlenecks
        self.inter_mode = inter_mode
        self.anchor_t = anchor_t
        self.dropout = dropout
        self.anchor_free = anchor_free
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        backbone = Backbone(self.feat_size, self.num_bottlenecks)

        obj_weight = 1.0 * (self.image_size / 640.0) ** 2
        box_weight = 0.05
        cls_weight = 0.5 * (self.num_classes / 80.0)

        if self.anchor_free:
            model = YoloAnchorFree(
                num_classes=self.num_classes,
                num_heads=self.num_heads,
                backbone=backbone,
                num_bottlenecks=self.num_bottlenecks,
                inter_mode=self.inter_mode,
                dropout=self.dropout
            )
            loss = YoloLossAnchorFree(
                num_classes=self.num_classes,
                strides=[head.stride for head in model.heads],
                obj_weight=obj_weight,
                box_weight=box_weight,
                cls_weight=cls_weight
            )
        else:
            model = Yolo(
                num_classes=self.num_classes,
                backbone=backbone,
                num_bottlenecks=self.num_bottlenecks,
                inter_mode=self.inter_mode,
                dropout=self.dropout
            )
            loss = YoloLoss(
                num_classes=self.num_classes,
                anchors=[head.anchors for head in model.heads],
                strides=[head.stride for head in model.heads],
                obj_weight=obj_weight,
                box_weight=box_weight,
                cls_weight=cls_weight,
                anchor_t=self.anchor_t,
            )
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)

    def forward(self, inputs, targets=None):
        """Forward
        """
        outputs = self.model(inputs)
        if targets is not None:
            losses = self.loss(outputs, targets)
            return losses
        else:
            return outputs
