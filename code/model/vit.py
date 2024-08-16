#!/usr/bin/env python3


import torch
from torch import nn
from timm.models import create_model


__all__ = [
    'Model'
]


class Model(nn.Module):

    def __init__(
            self,
            num_classes: int,
            backbone_name: str = 'vit_base_patch16_224',
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            separate_qkv: bool = True
    ) -> None:
        super(Model, self).__init__()
        self.model = Network(
            num_classes=num_classes,
            backbone_name=backbone_name,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            separate_qkv=separate_qkv
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets=None):
        y_ = self.model(inputs)
        if targets is None:
            return y_.argmax(-1)
        else:
            return self.loss(y_, targets)


class Network(nn.Module):

    def __init__(
            self,
            num_classes: int,
            backbone_name: str = 'vit_base_patch16_224',
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            separate_qkv: bool = False
    ) -> None:
        super(Network, self).__init__()
        self.backbone = create_model(
            backbone_name,
            checkpoint_path='model/ViT-B_16.npz',
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        if separate_qkv:
            self.separate_qkv()
        self.backbone.reset_classifier(num_classes)

    def separate_qkv(self):
        """only use to separate timm vit"""
        for blk in self.backbone.blocks:
            blk.attn.qkv = Separate_QKV(blk.attn.qkv)

    def forward(self, x):
        return self.backbone(x)


class Separate_QKV(nn.Module):
    def __init__(self, qkv: nn.Linear):
        super().__init__()
        dim = qkv.in_features
        if qkv.bias is not None:
            qkv_bias = True
        else:
            qkv_bias = False
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q.weight.data.copy_(qkv.weight.data[:dim, :])
        self.k.weight.data.copy_(qkv.weight.data[dim:-dim, :])
        self.v.weight.data.copy_(qkv.weight.data[-dim:, :])
        if qkv.bias is not None:
            self.q.bias.data.copy_(qkv.bias.data[:dim])
            self.k.bias.data.copy_(qkv.bias.data[dim:-dim])
            self.v.bias.data.copy_(qkv.bias.data[-dim:])

    def forward(self, x):
        x = torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)
        return x


def code_debuge():
    x = torch.rand((2 ,3, 224, 224))
    model = Model(num_classes=100, separate_qkv=True)
    y = model(x)
    for key, value in model.state_dict().items():
        print(key, value.shape)

if __name__ == '__main__':
    code_debuge()