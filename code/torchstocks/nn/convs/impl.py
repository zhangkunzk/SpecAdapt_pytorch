#!/usr/bin/env python3


import torch
from torch import nn

from torchstocks.nn.normalization import LayerNorm2d

__all__ = [
    'convs_84_4x32_64',
    'convs_128_5x32_64',
    'convs_128_5x64_128'
]


class Convs(nn.Module):

    def __init__(
            self,
            image_size: int,  # 96, 128, 224
            num_layers: int,  # 4
            ch_hid: int,  # 32, 64
            output_size: int  # 32, 64
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_layers = num_layers
        self.ch_hid = ch_hid
        self.output_size = output_size

        self.blocks = nn.Sequential()
        for i in range(self.num_layers):
            self.blocks.append(ConvBlock(3, self.ch_hid) if i == 0 else ConvBlock(self.ch_hid, self.ch_hid))
            image_size = image_size // 2
        self.flat_size = image_size * image_size * self.ch_hid

        self.fc = nn.Linear(self.flat_size, self.output_size)

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


def convs_84_4x32_64():
    return Convs(84, 4, 32, 64)


def convs_128_5x32_64():
    return Convs(128, 5, 32, 64)


def convs_128_5x64_128():
    return Convs(128, 5, 64, 128)
