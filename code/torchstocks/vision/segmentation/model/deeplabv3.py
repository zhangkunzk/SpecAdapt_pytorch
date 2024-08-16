#!/usr/bin/env python3

"""
@author: liying50
@since: 2022-11-08
"""

from typing import List, Dict, Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchstocks.nn.vision import ConvBlock2d
from torchstocks.vision.segmentation.model import resnet
from torchstocks.vision.segmentation.model.resnet import ResNet


__all__ = [
    "IntermediateLayerGetter",
    "FCN",
    "ASPP",
    "DeepLabV3",
    "Model"
]


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    def __init__(
            self,
            model: nn.Module,
            return_layers: Dict[str, str]
    ) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        """Forward
        """
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Sequential):
    """Fully Convolutional Networks
    """
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        modules = [
            ConvBlock2d(ch_in=in_channels, ch_out=inter_channels, kernel=3, padding=1, NonLin=nn.ReLU),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, kernel_size=1),
        ]
        super().__init__(*modules)


class ASPPConv(nn.Sequential):
    """Atrous Spatial Pyramid Pooling convolution module
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [ConvBlock2d(ch_in=in_channels, ch_out=out_channels, kernel=3,
                               padding=dilation, dilation=dilation, NonLin=nn.ReLU)]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Atrous Spatial Pyramid Pooling pooling module
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        modules = [
            nn.AdaptiveAvgPool2d(1),
            ConvBlock2d(ch_in=in_channels, ch_out=out_channels, kernel=1, NonLin=nn.ReLU)
        ]
        super().__init__(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        size = input.shape[-2:]
        for mod in self:
            input = mod(input)  # (b, c, 1, 1)
        return F.interpolate(input, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling
    """
    def __init__(
            self,
            in_channels: int,
            atrous_rates: List[int],
            out_channels: int = 256
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock2d(ch_in=in_channels, ch_out=out_channels, kernel=1, NonLin=nn.ReLU)
        self.asppconvs = []
        for atrous_rate in atrous_rates:
            self.asppconvs.append(ASPPConv(in_channels, out_channels, atrous_rate))
        self.aspppooling = ASPPPooling(in_channels, out_channels)
        self.convs = nn.ModuleList([self.conv1, *self.asppconvs, self.aspppooling])

        self.project = nn.Sequential(
            ConvBlock2d(ch_in=len(self.convs) * out_channels, ch_out=out_channels, kernel=1, NonLin=nn.ReLU),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class Head(nn.Sequential):
    """Head
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        modules = [
            ConvBlock2d(ch_in=in_channels, ch_out=256, kernel=3, padding=1, NonLin=nn.ReLU),
            nn.Conv2d(256, num_classes, kernel_size=1),
        ]
        super().__init__(*modules)


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "backbone_output" for the last feature map used, and "backbone_aux" if an
            auxiliary decoder is used.
        aspp (nn.Module): module that takes the "backbone_output" element returned from
            the backbone and returns multiscale features.
        decoder (nn.Module): module that takes the multiscale features returned from
            the aspp and returns a dense prediction.
        aux_decoder (nn.Module, optional): auxiliary decoder used during training
    """

    def __init__(
            self,
            backbone: nn.Module,
            aspp: nn.Module,
            decoder: nn.Module,
            aux_decoder: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.decoder = decoder
        self.aux_decoder = aux_decoder

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward
        """
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features['backbone_output']  # (b, c, h/8, w/8)
        x = self.aspp(x)  # (b, c', h/8, h/8)
        x = self.decoder(x)  # (b, num_classes, h/8, w/8)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result['output'] = x

        if self.aux_decoder is not None:
            x = features['backbone_aux']  # (b, c, h/8, w/8)
            x = self.aux_decoder(x)  # (b, num_classes, h/8, w/8)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result['aux'] = x

        return result


def _deeplabv3_resnet(
        backbone: ResNet,
        num_classes: int,
        aux: Optional[bool],
        atrous_rates=None,
        inner_channels=None
) -> DeepLabV3:
    return_layers = {'layer4': 'backbone_output'}
    if aux:
        return_layers['layer3'] = 'backbone_aux'
    if atrous_rates is None:
        atrous_rates = [12, 24, 36]
    if inner_channels is None:
        inner_channels = [2048, 1024]
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aspp = ASPP(inner_channels[0], atrous_rates, 256)
    aux_decoder = FCN(inner_channels[1], num_classes) if aux else None
    decoder = Head(256, num_classes)
    return DeepLabV3(backbone, aspp, decoder, aux_decoder)


class Model(nn.Module):
    """Define model
    """
    def __init__(
            self,
            num_classes: int,
            backbone: str = 'resnet50',
            output_stride: int = 8,
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
        else:
            inner_channels = [2048, 1024]

        backbone = backbone_fn(pretrained=pretrained, progress=True,
                               replace_stride_with_dilation=replace_stride_with_dilation)

        # Constructs a DeepLabV3 model with a ResNet backbone
        model = _deeplabv3_resnet(backbone, num_classes, aux_loss, atrous_rates, inner_channels)
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
