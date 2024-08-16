from typing import List

import torch
from torch import nn

__all__ = ["GeneralizedRCNN"]


class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(
            self,
            *,
            backbone: nn.Module,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
    ) -> None:
        """
        Args:
            backbone: a backbone module
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(
            self,
            batched_inputs: torch.Tensor,
            gt_instances: List[torch.Tensor]
    ):
        """
        Args:
            batched_inputs: (N, C, H, W)
            gt_instances: A list of N tensors, Tensor i has shape (k, 5), it contains k targets, and
                each target's format is (x1,y1,x2,y2,class).
        Returns:
            if train:
                return Dict[str, torch.Tensor]
            if inference:
                return list[torch.Tensor]:
                Each Tensor is the output for one input image.
                The Tensor's shape is (k, 6), it contains k objects, each object's format is
                (x1,y1,x2,y2,pred_class,score).
        """
        if not self.training:
            return self.inference(batched_inputs)
        features = self.backbone(batched_inputs)
        image_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in batched_inputs]
        assert self.proposal_generator is not None
        proposals, proposal_losses = self.proposal_generator(image_sizes, features, gt_instances)
        _, detector_losses = self.roi_heads(features, proposals, gt_instances)
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs: torch.Tensor):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (torch.Tensor): same as in :meth:`forward`
        Returns:
            a list[dict] containing raw network outputs.
        """
        assert not self.training
        features = self.backbone(batched_inputs)
        image_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in batched_inputs]  # h,w
        assert self.proposal_generator is not None
        proposals, _ = self.proposal_generator(image_sizes, features, None)
        for _index, _ in enumerate(proposals):
            proposals[_index]['image_size'] = image_sizes[_index]
        results = self.roi_heads(features, proposals, None)
        return results
