#!/usr/bin/env python3

from torch import nn

from torchstocks import nn as nn_
from torchstocks.models.adapters import ResnetAdapter
from torchstocks.utils import import_by_name

__all__ = [
    'Model'
]


class Model(nn.Module):
    """Define model
    """

    def __init__(
            self,
            backbone: str,
            num_classes: int = None,
            non_lin: str = None,
            norm: str = None,
            pretrained: bool = False
    ) -> None:
        super(Model, self).__init__()
        self.model = Network(
            backbone=backbone,
            non_lin=non_lin,
            norm=norm,
            num_classes=num_classes,
            pretrained=pretrained
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets=None):
        """Forward
        """
        y_ = self.model(inputs)
        if targets is None:
            return y_.softmax(-1)
        else:
            return self.loss(y_, targets)


class Network(nn.Module):
    """Network
    """

    def __init__(
            self,
            backbone: str,
            num_classes: int,
            non_lin: str = None,
            norm: str = None,
            pretrained: bool = False
    ) -> None:
        super(Network, self).__init__()
        Backbone = import_by_name(backbone)

        NonLin = None
        if non_lin is not None:
            for n in [nn, nn_]:
                NonLin = getattr(n, non_lin, None)
                if NonLin is not None:
                    break

        Norm = None
        if norm is not None:
            for n in [nn, nn_]:
                Norm = getattr(n, norm, None)
                if Norm is not None:
                    break

        if pretrained:
            net = Backbone(True)
            self.backbone = ResnetAdapter(net)
            self.fc = net.fc
        else:
            self.backbone = ResnetAdapter(Backbone(pretrained))
            _replace_modules(self.backbone, NonLin, Norm)
            self.fc = nn.Linear(self.backbone.ch_out_list[-1], num_classes)

    def forward(self, x):
        """Forward
        """
        h = self.backbone(x)[-1]
        h = h.mean((2, 3))
        return self.fc(h)


def _replace_modules(module: nn.Module, NonLin, Norm):
    for name, child in module.named_children():
        if NonLin is not None and isinstance(child, nn.ReLU):
            setattr(module, name, NonLin())
        if Norm is not None and isinstance(child, nn.BatchNorm2d):
            if issubclass(Norm, nn.GroupNorm):
                norm = Norm(32, child.num_features)
            else:
                norm = Norm(child.num_features)
            setattr(module, name, norm)
        _replace_modules(child, NonLin, Norm)
