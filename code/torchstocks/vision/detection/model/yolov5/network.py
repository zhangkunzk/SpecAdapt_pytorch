#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-12-16
"""

import math
from typing import List, Union, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from torchstocks.nn.vision import ConvBlock2d, CSP2d, FastSPP2d, PACPyramidNetwork2d

DEFAULT_ANCHORS = np.array([
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)]
], dtype=np.float32)


class Backbone(nn.Module):
    """Backbone
    """

    def __init__(
            self,
            feat_size: int = 64,
            num_bottlenecks: int = 3
    ) -> None:
        super(Backbone, self).__init__()
        self.size0 = feat_size * 2
        self.layer0 = nn.Sequential(
            ConvBlock2d(3, feat_size, 6, 2, 2),
            ConvBlock2d(feat_size, self.size0, 3, 2),
            CSP2d(self.size0, self.size0, num_bottlenecks)
        )
        self.size1 = feat_size * 4
        self.layer1 = nn.Sequential(
            ConvBlock2d(self.size0, self.size1, 3, 2),
            CSP2d(self.size1, self.size1, num_bottlenecks * 2)
        )
        self.size2 = feat_size * 8
        self.layer2 = nn.Sequential(
            ConvBlock2d(self.size1, self.size2, 3, 2),
            CSP2d(self.size2, self.size2, num_bottlenecks * 3)
        )
        self.size3 = feat_size * 16
        self.layer3 = nn.Sequential(
            ConvBlock2d(self.size2, self.size3, 3, 2),
            CSP2d(self.size3, self.size3, num_bottlenecks),
            FastSPP2d(self.size3, self.size3, 5)
        )
        self.output_sizes = [self.size0, self.size1, self.size2, self.size3]
        self.strides = [4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward
        """
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        return [h0, h1, h2, h3]


class ResnetAdapter(nn.Module):
    """Resnet adapter
    """

    def __init__(self, model: nn.Module) -> None:
        super(ResnetAdapter, self).__init__()
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        # the following code will inference the net to get feature size
        # setting the backbone to eval mode to prevent its BN layers from being corrupted
        self.eval()
        self.output_sizes = [
            h.shape[1]
            for h in self(torch.rand((1, 3, 64, 64), dtype=torch.float32))
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward
        """
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        return [h0, h1, h2, h3]


class FPN2d(PACPyramidNetwork2d):
    """Feature Pyramid Network
    """

    def __init__(
            self,
            ch_in_list: Sequence[int],
            ch_out_list: Sequence[int],
            depth: int,
            inter_mode='nearest',
            dropout=0.0,
            Norm=None,
            NonLin=None
    ) -> None:
        super(FPN2d, self).__init__(
            ch_in_list=ch_in_list,
            ch_hid_list=ch_out_list,
            ch_out_list=ch_out_list,
            depth=depth,
            inter_mode=inter_mode,
            dropout=dropout,
            Norm=Norm,
            NonLin=NonLin,
        )


class PAN2d(PACPyramidNetwork2d):
    """Path Aggregation Network
    """

    def __init__(
            self,
            ch_in_list: Sequence[int],
            ch_out_list: Sequence[int],
            depth: int,
            inter_mode='nearest',
            dropout=0.0,
            Norm=None,
            NonLin=None
    ) -> None:
        super(PAN2d, self).__init__(
            ch_in_list=ch_in_list,
            ch_hid_list=ch_in_list,
            ch_out_list=ch_out_list,
            depth=depth,
            inter_mode=inter_mode,
            dropout=dropout,
            Norm=Norm,
            NonLin=NonLin,
        )


class Head(nn.Module):
    """Define head
    """

    def __init__(
            self,
            ch_in: int,
            num_classes: int,
            anchors: torch.Tensor,
            stride: int
    ) -> None:
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.stride = stride
        self.num_anchors = len(self.anchors)

        ch_out = (self.num_classes + 5) * self.num_anchors
        self.conv = nn.Conv2d(ch_in, ch_out, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        y = self.conv(x)

        # n (a d) h w -> n a h w d
        n, _, h, w = y.shape
        y = y.reshape((n, self.num_anchors, -1, h, w))
        y = y.permute((0, 1, 3, 4, 2)).contiguous()

        xy = y[..., 0:2].sigmoid() * 2.0 - 0.5  # cell
        wh = (y[..., 2:4].sigmoid() * 2.0).square() * (self.anchors[:, None, None, :] / self.stride)  # cell

        if self.training:
            obj_cls = y[..., 4:]
            y = torch.cat([xy, wh, obj_cls], -1)
        else:
            y_axis = torch.arange(h, device=y.device)
            x_axis = torch.arange(w, device=y.device)
            grid = torch.stack([x_axis[None, :].repeat((h, 1)), y_axis[:, None].repeat((1, w))], -1)
            gain = torch.tensor([1.0 / w, 1.0 / h], dtype=torch.float32, device=y.device)
            xy = (xy + grid) * gain  # pct
            wh = wh * gain  # pct
            obj_cls = y[..., 4:].sigmoid()
            y = torch.cat([xy, wh, obj_cls], -1)
            y = y.view(n, -1, (self.num_classes + 5))  # for the convenience of convert model to onnx

        return y


class HeadAnchorFree(nn.Module):
    """Define head for anchor-free method
    """

    def __init__(
            self,
            ch_in: int,
            num_classes: int,
            stride
    ) -> None:
        super(HeadAnchorFree, self).__init__()
        self.ch_in = ch_in
        self.num_classes = num_classes
        self.stride = stride

        self.conv = nn.Conv2d(ch_in, self.num_classes + 5, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forawrd
        """
        y = self.conv(x)
        y = y.permute((0, 2, 3, 1)).contiguous()

        xy = y[..., 0:2].sigmoid() * 2.0 - 0.5  # cell
        # wh = y[..., 2:4].sigmoid().mul(2).square() * 8  # cell
        wh = y[..., 2:4].mul(0.5).exp().mul(5.0)  # cell

        h, w = x.shape[2], x.shape[3]
        if self.training:
            obj_cls = y[..., 4:]
            y = torch.concat([xy, wh, obj_cls], -1)
        else:
            y_axis = torch.arange(h, device=y.device)
            x_axis = torch.arange(w, device=y.device)
            grid = torch.stack([x_axis[None, :].repeat((h, 1)), y_axis[:, None].repeat((1, w))], -1)
            gain = torch.tensor([1.0 / w, 1.0 / h], dtype=torch.float32, device=y.device)
            xy = (xy + grid) * gain  # pct
            wh = wh * gain  # pct
            obj_cls = y[..., 4:].sigmoid()
            y = torch.concat([xy, wh, obj_cls], -1)
            y = y.view(x.shape[0], -1, (self.num_classes + 5))  # for the convenience of convert model to onnx

        return y


class DecoupledHead(nn.Module):
    """Define decoupled head
    """

    def __init__(
            self,
            ch_in: int,
            num_classes: int,
            stride: int,
            ch_hid: int = 32,
            kernel_size: int = 3,
            Norm=None,
            NonLin=None
    ) -> None:
        super(DecoupledHead, self).__init__()
        self.ch_in = ch_in
        self.num_classes = num_classes
        self.stride = stride
        self.ch_hid = ch_hid
        self.kernel_size = kernel_size
        self.Norm = Norm
        self.NonLin = NonLin

        self.conv_box = nn.Sequential(
            ConvBlock2d(self.ch_in, self.ch_hid, self.kernel_size, Norm=self.Norm, NonLin=self.NonLin),
            nn.Conv2d(ch_hid, 4, (1, 1), (1, 1), (0, 0), bias=True)
        )
        self.conv_obj = nn.Sequential(
            ConvBlock2d(self.ch_in, self.ch_hid, self.kernel_size, Norm=self.Norm, NonLin=self.NonLin),
            nn.Conv2d(ch_hid, 1, (1, 1), (1, 1), (0, 0), bias=True)
        )
        self.conv_cls = nn.Sequential(
            ConvBlock2d(self.ch_in, self.ch_hid, self.kernel_size, Norm=self.Norm, NonLin=self.NonLin),
            nn.Conv2d(ch_hid, self.num_classes, (1, 1), (1, 1), (0, 0), bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        box, obj, cls = map(
            lambda _f: _f(x).permute((0, 2, 3, 1)).contiguous(),
            [self.conv_box, self.conv_obj, self.conv_cls]
        )

        xy = box[..., 0:2].sigmoid() * 2.0 - 0.5  # cell
        # wh = box[..., 2:4].sigmoid().mul(2).square() * 8  # cell
        wh = box[..., 2:4].mul(0.5).exp().mul(5.0)  # cell

        h, w = x.shape[2], x.shape[3]
        if self.training:
            y = torch.concat([xy, wh, obj, cls], -1)
        else:
            y_axis = torch.arange(h, device=x.device)
            x_axis = torch.arange(w, device=x.device)
            grid = torch.stack([x_axis[None, :].repeat((h, 1)), y_axis[:, None].repeat((1, w))], -1)
            gain = torch.tensor([1.0 / w, 1.0 / h], dtype=torch.float32, device=x.device)
            xy = (xy + grid) * gain  # pct
            wh = wh * gain  # pct
            obj = obj.sigmoid()
            cls = cls.sigmoid()
            y = torch.concat([xy, wh, obj, cls], -1)
            y = y.view(x.shape[0], -1, (self.num_classes + 5))  # for the convenience of convert model to onnx

        return y


def test_backbone(
        backbone: nn.Module,
        image_size: int = 640
) -> Tuple[Sequence[int], Sequence[int]]:
    """ Test backbone to get some parameters
    """
    dummy_image = torch.rand((1, 3, image_size, image_size), dtype=torch.float32)

    mode = backbone.training
    backbone.eval()
    outputs = backbone(dummy_image)
    backbone.train(mode)

    assert isinstance(outputs, (list, tuple))
    output_sizes, strides = [], []
    for output in outputs:
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        _, c, h, w = output.shape
        assert h == w
        output_sizes.append(int(c))
        strides.append(math.ceil(image_size / h))
    return output_sizes, strides


class Yolo(nn.Module):
    """Define yolo model
    """

    def __init__(
            self,
            *,
            num_classes: int,
            anchors: Union[torch.Tensor, np.ndarray, List] = DEFAULT_ANCHORS,
            backbone,
            num_bottlenecks=3,
            inter_mode='nearest',
            dropout=0.0
    ) -> None:
        super(Yolo, self).__init__()
        assert num_bottlenecks >= 1
        self.num_classes = num_classes

        if not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors)
        assert len(anchors.shape) == 3 and anchors.shape[-1] == 2
        assert anchors.shape[0] <= 3

        self.num_heads = len(anchors)
        assert self.num_heads <= 3

        self.backbone = backbone
        backbone_sizes, strides = test_backbone(backbone)
        self.backbone_sizes = backbone_sizes[-self.num_heads:]
        self.strides = strides[-self.num_heads:]

        in_chs = list(reversed(self.backbone_sizes))
        out_chs = [int(ch * 0.5) for ch in in_chs]
        self.fpn = FPN2d(in_chs, out_chs, depth=num_bottlenecks * 2, inter_mode=inter_mode, dropout=dropout)
        self.fpn_sizes = out_chs
        in_chs = list(reversed(self.fpn_sizes))
        out_chs = list(self.backbone_sizes)
        self.pan = PAN2d(in_chs, out_chs, depth=num_bottlenecks * 2, inter_mode=inter_mode, dropout=dropout)
        self.pan_sizes = out_chs

        # self.fpn = FPN(list(reversed(self.backbone_sizes)), num_bottlenecks, inter_mode=inter_mode)
        # self.fpn_sizes = self.fpn.output_sizes
        # self.pan = PAN(list(reversed(self.fpn_sizes)), num_bottlenecks, inter_mode=inter_mode)
        # self.pan_sizes = self.pan.output_sizes

        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            head = Head(self.pan_sizes[i], num_classes, anchors[i], self.strides[i])
            self.heads.append(head)

    def forward(self, x: torch.Tensor):
        """Forward
        """
        feat_maps = self.backbone(x)[-self.num_heads:]
        feat_maps.reverse()
        feat_maps = self.fpn(feat_maps)
        feat_maps.reverse()
        feat_maps = self.pan(feat_maps)
        outputs = list(map(lambda _a: _a[1](_a[0]), zip(feat_maps, self.heads)))
        if not self.training:
            outputs = torch.cat(outputs, 1)  # for the convenience of convert model to onnx
        return outputs


class YoloAnchorFree(nn.Module):
    """Define anchor-free yolo model
    """

    def __init__(
            self,
            *,
            num_classes: int,
            num_heads: int,
            backbone,
            num_bottlenecks=3,
            inter_mode='nearest',
            dropout=0.0
    ) -> None:
        super(YoloAnchorFree, self).__init__()
        assert num_bottlenecks >= 1
        self.num_classes = num_classes

        self.num_heads = num_heads
        assert self.num_heads <= 3

        self.backbone = backbone
        backbone_sizes, strides = test_backbone(backbone)
        self.backbone_sizes = backbone_sizes[-self.num_heads:]
        self.strides = strides[-self.num_heads:]

        in_chs = list(reversed(self.backbone_sizes))
        out_chs = [int(ch * 0.5) for ch in in_chs]
        self.fpn = FPN2d(in_chs, out_chs, depth=num_bottlenecks * 2, inter_mode=inter_mode, dropout=dropout)
        self.fpn_sizes = out_chs
        in_chs = list(reversed(self.fpn_sizes))
        out_chs = list(self.backbone_sizes)
        self.pan = PAN2d(in_chs, out_chs, depth=num_bottlenecks * 2, inter_mode=inter_mode, dropout=dropout)
        self.pan_sizes = out_chs

        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            # head = DecoupledHead(self.pan_sizes[i], num_classes, self.strides[i])
            head = HeadAnchorFree(self.pan_sizes[i], num_classes, self.strides[i])
            self.heads.append(head)

    def forward(self, x: torch.Tensor):
        """Forward
        """
        feat_maps = self.backbone(x)[-self.num_heads:]
        feat_maps.reverse()
        feat_maps = self.fpn(feat_maps)
        feat_maps.reverse()
        feat_maps = self.pan(feat_maps)
        outputs = list(map(lambda _a: _a[1](_a[0]), zip(feat_maps, self.heads)))
        if not self.training:
            outputs = torch.cat(outputs, 1)  # for the convenience of convert model to onnx
        return outputs
