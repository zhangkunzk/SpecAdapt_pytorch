#!/usr/bin/env python3


from typing import Mapping, Any, Sequence

from torch import nn, Tensor
from torch.nn import functional as F

from torchstocks.utils import import_by_name

__all__ = [
    'ClassificationModel'
]


class ClassificationModel(nn.Module):

    def __init__(
            self,
            backbone: str,
            num_classes: int,
            backbone_kwargs: Mapping[str, Any] = None,
            ce_eps: float = 1e-10
    ) -> None:
        super().__init__()
        create_backbone = import_by_name(backbone)
        if backbone_kwargs is None:
            backbone_kwargs = {}
        self.backbone = create_backbone(**backbone_kwargs)

        output_size = None
        if output_size is None:
            output_size = getattr(self.backbone, 'ch_out', None)
        if output_size is None:
            output_size = getattr(self.backbone, 'ch_out_list', None)
        if output_size is None:
            output_size = getattr(self.backbone, 'output_size', None)
        if output_size is None:
            raise RuntimeError('Cannot get the feature dimension of the backbone.')
        if isinstance(output_size, Sequence):
            output_size = output_size[-1]
        self.head = nn.Linear(output_size, num_classes)
        self.num_classes = num_classes
        self.ce_eps = ce_eps

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        h = self.backbone(x)
        if isinstance(h, Sequence):
            h = h[-1]
        r = len(h.shape)
        if r > 2:
            h = h.mean([i for i in range(2, r)])
        logit = self.head(h)

        if y is None:
            return logit
        else:
            prob = F.softmax(logit, -1)
            target = F.one_hot(y, self.num_classes).float()
            loss = ((target * prob).sum(-1) + self.ce_eps).log().mean().neg()
            return loss
