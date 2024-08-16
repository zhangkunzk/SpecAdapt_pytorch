#!/usr/bin/env python3


from typing import Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F

from torchstocks.nn.normalization import LayerNorm2d


class FPNAdapter(nn.Module):
    """fpn adapter
    Args:
        embed_size (int): transformer embeded dims
        scale_factor_list (Sequence[float]): feature scale. 0: nothing to do.
    """

    def __init__(
            self,
            embed_size: int = 768,
            scale_factor_list: Sequence[int] = (2., 1., 0.5)
    ) -> None:
        super().__init__()
        self.stage_layers = MuitiScaleConvList(scale_factor_list=scale_factor_list, in_channels=embed_size)
        self.fuse_layers = FuseBlock(self.stage_layers.out_channel_list)

    def forward(self, input_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        temp_list = []
        for index, layer in enumerate(self.stage_layers):
            temp_list.append(layer(input_list[index]))
        return self.fuse_layers(temp_list)


class MuitiScaleConvList(nn.ModuleList):
    """"consturct multi-scale conv list"""

    def __init__(
            self,
            scale_factor_list: Sequence[int] = (2., 1., 0.5),
            in_channels: int = 768
    ) -> None:
        multi_scale_conv_list = []
        self.out_channel_list = []
        for scale in scale_factor_list:
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                    __init__(in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=False),
                    __init__(in_channels // 4),
                    nn.GELU()
                ]
                self.out_channel_list.append(in_channels // 4)
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                    __init__(in_channels // 2),
                    nn.GELU()
                ]
                self.out_channel_list.append(in_channels // 2)
            elif scale == 1.0:
                layers = [
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                    __init__(in_channels),
                    nn.GELU()
                ]
                self.out_channel_list.append(in_channels)
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                self.out_channel_list.append(in_channels)
            elif scale == 0.:
                layers = [nn.Identity()]  # nothing to do
                self.out_channel_list.append(in_channels)
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            multi_scale_conv_list.append(nn.Sequential(*layers))
        super().__init__(multi_scale_conv_list)




class FuseBlock(nn.Module):
    """Fuse feature list by concat like fpn
    feature sequence: image size from large to small, from shallow to deep
    """

    def __init__(
            self,
            ch_list: Sequence[int],
            mode: str = 'bilinear',
            NormType=LayerNorm2d,
            ActType=nn.GELU
    ) -> None:
        super().__init__()
        self.layer_list = nn.ModuleList()
        self.up_list = nn.ModuleList()

        assert len(ch_list) >= 2, 'Two few feature maps to fuse.'
        for i in range(len(ch_list) - 2, -1, -1):
            ch_in = ch_list[i + 1] + ch_list[i]
            ch_out = ch_list[i]
            self.layer_list.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
                NormType(ch_out),
                ActType()
            ))
            # self.up_list.append(
            #     nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2, bias=False)
            #     if mode == 'conv' else
            #     nn.Sequential(
            #         nn.UpsamplingBilinear2d(scale_factor=2),
            #         nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
            #     )
            # )

    def forward(self, feature_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """feature sequence: image size from large to small, from shallow to deep"""
        output_list = [feature_list[-1]]
        for feature, up, layer in zip(reversed(feature_list[:-1]), self.up_list, self.layer_list):
            up_feature = up(output_list[-1])
            cat_feature = torch.concat([up_feature, feature], 1)
            merged_feature = layer(cat_feature)
            output_list.append(merged_feature)
        return reversed(output_list)
