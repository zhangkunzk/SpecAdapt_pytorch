#!/usr/bin/env python3


import torch
from torch import nn

from torchstocks.nn.normalization import LayerNorm2d

__all__ = [
    'convs_96_4x32_32',
    'convs_84_4x32_64'
]


class SimpleConvs(nn.Module):

    def __init__(
            self,
            image_size: int,  # 96, 128, 224
            ch_hid: int,  # 32, 64
            num_layers: int,  # 4
            output_size: int  # 32, 64
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.append(ConvBlock(3, ch_hid) if i == 0 else ConvBlock(ch_hid, ch_hid))
            image_size = image_size // 2
        self.flat_size = image_size * image_size * ch_hid

        self.fc = nn.Linear(self.flat_size, output_size)

    def forward(self, x: torch.Tensor):
        h = self.blocks(x)
        h = h.reshape((h.shape[0], -1))
        h = self.fc(h)
        return h


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_norm=True, use_act=True):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            LayerNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.GELU() if use_act else nn.Identity(),
            nn.MaxPool2d((2, 2), (2, 2))
        )


def convs_96_4x32_32():
    return SimpleConvs(96, 32, 4, 32)


def convs_84_4x32_64():
    return SimpleConvs(84, 32, 4, 64)
