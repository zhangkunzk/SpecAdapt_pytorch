#!/usr/bin/env python3

"""
@author: liying50
@since: 2022-11-15
"""

from typing import Dict, Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchstocks.vision.segmentation.model import resnet
from torchstocks.vision.segmentation.model.resnet import ResNet
from torchstocks.vision.segmentation.model.deeplabv3 import IntermediateLayerGetter, FCN, ASPP, Head
from torchstocks.nn.vision import ConvBlock2d

__all__ = [
    "DeepLabV3Plus",
    "Model"
]


class DeepLabV3Plus(nn.Module):
    """
    Implements DeepLabV3Plus model.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "backbone_output" for the last feature map used, "lowlevel_feature" for the
            lowlevel feature map used, and "backbone_aux" if an auxiliary decoder is used.
        aspp (nn.Module): module that takes the "backbone_output" element returned from
            the backbone and returns multiscale features.
        decoder (nn.Module): module that takes the "backbone_output" element returned from
            the backbone and returns a dense prediction.
        aux_decoder (nn.Module, optional): auxiliary decoder used during training
    """

    def __init__(
            self,
            backbone: nn.Module,
            aspp: nn.Module,
            lowlevel_channels: int,
            decoder: nn.Module,
            aux_decoder: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.decoder = decoder
        self.aux_decoder = aux_decoder
        self.shortcut = ConvBlock2d(ch_in=lowlevel_channels, ch_out=256, kernel=1, NonLin=nn.ReLU)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward
        """
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        lowlevel_feature = features['lowlevel_feature']  # (b, c, h/4, w/4)
        lowlevel_feature_shape = lowlevel_feature.shape[-2:]
        x = features['backbone_output']  # (b, c, h/16, w/16)
        aspp_out = self.aspp(x)
        aspp_out = F.interpolate(aspp_out, size=lowlevel_feature_shape, mode='bilinear', align_corners=True)
        shortcut_out = self.shortcut(lowlevel_feature)
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        x = self.decoder(feats)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result['output'] = x

        if self.aux_decoder is not None:
            x = features['backbone_aux']  # (b, c, h/16, w/16)
            x = self.aux_decoder(x)  # (b, num_classes, h/16, w/16)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result['aux'] = x

        return result


def _deeplabv3plus_resnet(
        backbone: ResNet,
        num_classes: int,
        aux: Optional[bool],
        lowlevel_channels: int = 256,
        atrous_rates=None,
        inner_channels=None
) -> DeepLabV3Plus:
    """Deeplabv3plus resnet
    """
    if atrous_rates is None:
        atrous_rates = [12, 24, 36]
    if inner_channels is None:
        inner_channels = [2048, 1024]
    return_layers = {'layer1': 'lowlevel_feature', 'layer4': 'backbone_output'}
    if aux:
        return_layers['layer3'] = 'backbone_aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aspp = ASPP(inner_channels[0], atrous_rates, 256)
    aux_decoder = FCN(inner_channels[1], num_classes) if aux else None
    decoder = Head(256 * 2, num_classes)
    return DeepLabV3Plus(backbone, aspp, lowlevel_channels, decoder, aux_decoder)


class Model(nn.Module):
    """Define model
    """
    def __init__(
            self,
            num_classes: int,
            backbone: str = 'resnet50',
            output_stride: int = 16,
            pretrained: Optional[bool] = True,
            aux_loss: Optional[bool] = True
    ) -> None:
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.aux_loss = aux_loss
        self.loss = None

        backbone_fn = getattr(resnet, self.backbone_name, None)
        print(self.backbone_name)

        if output_stride == 16:
            atrous_rates = [6, 12, 18]
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
            replace_stride_with_dilation = [False, True, True]
        else:
            raise NotImplementedError
        if self.backbone_name in ['resnet18', 'resnet34']:
            inner_channels = [512, 256]
            lowlevel_channels = 64
        else:
            inner_channels = [2048, 1024]
            lowlevel_channels = 256

        backbone = backbone_fn(pretrained=pretrained, progress=True,
                               replace_stride_with_dilation=replace_stride_with_dilation)

        # Constructs a DeepLabV3Plus model with a ResNet backbone
        model = _deeplabv3plus_resnet(
            backbone, num_classes, aux_loss, lowlevel_channels, atrous_rates, inner_channels
        )
        loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)

    def forward(self, inputs: Tensor, targets=None):
        """Forward
        """
        n, _, _, _ = inputs.shape
        outputs = self.model(inputs)
        if targets is None:
            output = torch.argmax(outputs['output'], dim=1)
            return output
        else:
            _loss = self.loss(outputs['output'], targets)
            if self.aux_loss:
                _aux_loss = self.loss(outputs['aux'], targets)
                _loss += 0.4 * _aux_loss
            return _loss / n
