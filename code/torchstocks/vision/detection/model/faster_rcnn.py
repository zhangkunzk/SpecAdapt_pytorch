#!/usr/bin/env python3

"""
@author: liying50
@since: 2023-03-01
"""

import torch
from torch import nn
from torchvision import models

from torchstocks.vision.detection.model.rcnn.backbone import ResnetAdapter
from torchstocks.vision.detection.model.rcnn.poolers import ROIPooler
from torchstocks.vision.detection.model.rcnn.rpn import build_proposal_generator
from torchstocks.vision.detection.model.rcnn.roi_heads import build_roi_heads
from torchstocks.vision.detection.model.rcnn.rcnn import GeneralizedRCNN


class Model(nn.Module):
    """Define model
    """

    def __init__(
            self,
            num_classes: int,
            backbone: str,
            pooler_resolution: int = 14,
            box_reg_loss_type: str = 'smooth_l1',
            batch_size_per_image: int = 512,
            in_features=None
    ) -> None:
        super(Model, self).__init__()
        if in_features is None:
            in_features = ['res4']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone_name = backbone
        self.in_features = in_features
        backbone_fn = getattr(models, self.backbone_name, None)
        backbone = ResnetAdapter(backbone_fn(True))
        input_shape = backbone.output_shape()
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales
        )
        res5 = backbone.layer4
        res5_out_channels = input_shape['res5'].channels
        proposal_generator = build_proposal_generator(
            in_features=in_features,
            input_shape=input_shape,
            rpn_box_reg_loss_type=box_reg_loss_type
        )
        roi_heads = build_roi_heads(
            res5=res5,
            res5_out_channels=res5_out_channels,
            in_features=in_features,
            pooler=pooler,
            num_classes=num_classes,
            batch_size_per_image=batch_size_per_image,
            box_reg_loss_type=box_reg_loss_type
        )
        model = GeneralizedRCNN(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads
        )
        self.model = model.to(self.device)

    def forward(self, inputs, targets=None):
        """Forward
        """
        outputs = self.model(batched_inputs=inputs, gt_instances=targets)
        return outputs
