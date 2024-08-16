#!/usr/bin/env python3

"""
@author: Yubin
@since: 2023-5-15
"""

from typing import Sequence
import torch
from torch import nn

from torchstocks.nn.vit.sam_vit import network
from torchstocks.nn.normalization import LayerNorm2d


class MultiScaleSamVitAdapter(nn.Module):
    """multi-scaled vit adapter, add conv in every stage
    backbone_name: backbone name
    scale_factors (Sequence[float]): scale feature from backbone. Because sam vit has neck block,
    we ignore the feature of vit last stage(output for neck block).
    stages (Sequence[int]): the stage index for taking part in fuse.
    """
    def __init__(
            self,
            backbone_name: str = 'vit_base',
            scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0),
            stages: Sequence[int] = (0, 1, 2, 3),
            pretrained: bool = True
    ) -> None:
        super().__init__()
        assert len(scale_factors) == len(stages), 'stages is not equal to factors!'
        self.ch_out_list = []
        self.stride_list = []
        self.stages = stages

        self.vit = getattr(network, backbone_name)(pretrained=pretrained)
        embed_size = self.vit.embed_size
        self.fpn_block = FPNBlock(embed_size=embed_size, scale_factor_list=scale_factors)

        state = self.training
        self.train(False)
        for output in self(torch.rand((1, 3, 512, 512), dtype=torch.float32)):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            _, c, h, w = output.shape
            self.ch_out_list.append(int(c))
            self.stride_list.append(int(512 / h))
        self.train(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_feature_list = self.vit(x)
        fuse_feature_list = [
            stage_feature_list[i] for i in range(len(stage_feature_list))
            if i in self.stages
        ]
        output = self.fpn_block(fuse_feature_list)
        return output


class FPNBlock(nn.Module):
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
        self.stage_layers = MultiScaleConvList(scale_factor_list=scale_factor_list, in_channels=embed_size)
        self.fuse_layers = FuseMultiFeatureBlock(self.stage_layers.out_channel_list)

    def forward(self, input_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        temp_list = []
        for index, layer in enumerate(self.stage_layers):
            temp_list.append(layer(input_list[index]))
        return self.fuse_layers(temp_list)


class MultiScaleConvList(nn.ModuleList):
    """"construct multi-scale conv list"""
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
                    LayerNorm2d(in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, bias=False),
                    LayerNorm2d(in_channels // 4),
                    nn.GELU()
                ]
                self.out_channel_list.append(in_channels // 4)
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False),
                    LayerNorm2d(in_channels // 2),
                    nn.GELU()
                ]
                self.out_channel_list.append(in_channels // 2)
            elif scale == 1.0:
                layers = [
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                    LayerNorm2d(in_channels),
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


class FuseMultiFeatureBlock(nn.Module):
    """Fuse feature list by concat like fpn
    feature sequence: image size from large to small, from shallow to deep
    """
    def __init__(
            self,
            in_channel_list: Sequence[int],
            up_mode: str = 'bilinear'
    ) -> None:
        super().__init__()
        assert len(in_channel_list) >= 2
        self.layer_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        in_channel_list.reverse()
        # reverse channel list, from deep feature to shallow
        for index, in_channel in enumerate(in_channel_list):
            if index == len(in_channel_list) - 1:
                break
            layer = nn.Sequential(
                nn.Conv2d(in_channel + in_channel_list[index + 1], in_channel_list[index + 1],
                          kernel_size=1, stride=1, bias=False),
                LayerNorm2d(in_channel_list[index + 1]),
                nn.GELU()
            )
            self.layer_list.append(layer)
            self.up_list.append(
                nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2, bias=False)
                if up_mode == 'conv' else
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
                )
            )

    def forward(self, feature_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """feature sequence: image size from large to small, from shallow to deep"""
        output = []
        # reverse feature list, from deep feature to shallow
        feature_list.reverse()
        output.append(feature_list[0])
        for index, feature in enumerate(feature_list[1:]):
            up_feature = self.up_list[index](output[-1])
            temp = torch.concat([up_feature, feature], dim=1)
            merge_feature = self.layer_list[index](temp)
            output.append(merge_feature)
        # from shallow to deep
        output.reverse()
        return output

################################################################################
# Code for debug  and test.
################################################################################

def test():
    x = torch.randn(size=(2, 3, 640, 640))
    model = MultiScaleSamVitAdapter(backbone_name='vit_base', scale_factors=(2, 1.0, 0.5), stages=(1, 2, 3))
    y = model(x)
    for j in y:
        print(j.shape)
    print(model.ch_out_list, model.stride_list)
    return 0


if __name__ == '__main__':
    exit(test())
