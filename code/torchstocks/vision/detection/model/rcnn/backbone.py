from typing import Dict

import torch
from torch import nn

from .utils import ShapeSpec


class ResnetAdapter(nn.Module):
    """Resenet adapter
    """

    def __init__(self, model: nn.Module) -> None:
        super(ResnetAdapter, self).__init__()
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # the following code will inference the net to get feature size
        # setting the backbone to eval mode to prevent its BN layers from being corrupted
        self.eval()
        self.strides = {k: 128 / v.shape[2] for k, v in self(torch.rand((1, 3, 128, 128), dtype=torch.float32)).items()}
        self.out_feature_channels = {k: v.shape[1] for k, v in
                                     self(torch.rand((1, 3, 128, 128), dtype=torch.float32)).items()}
        self.out_features = ["res1", "res2", "res3", "res4", "res5"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward
        """
        res1 = self.layer0(x)
        res2 = self.layer1(res1)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        return {"res1": res1, "res2": res2, "res3": res3, "res4": res4, "res5": res5}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self.out_feature_channels[name], stride=self.strides[name]
            )
            for name in self.out_features
        }
