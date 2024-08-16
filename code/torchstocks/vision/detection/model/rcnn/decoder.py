#!/usr/bin/env python3

"""
@author: liying50
@since: 2022-07-11
"""

from typing import Dict, List, Tuple

import torch
import torchvision
from torch import nn

from .utils import boxes_clip


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
        topk_per_image: int,
) -> Dict[str, torch.Tensor]:
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    """
    valid_mask = torch.isfinite(boxes).all(
        dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    box_dim = 4
    num_bbox_reg_classes = boxes.shape[1] // box_dim
    boxes = boxes.reshape(-1, box_dim)
    boxes = boxes_clip(boxes, image_shape)
    boxes = boxes.view(-1, num_bbox_reg_classes, box_dim)  # (R, C, 4)

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_threshold  # (R, K)
    # R' * 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    # 2. Apply NMS for each class independently.
    keep = torchvision.ops.boxes.batched_nms(
        boxes.float(), scores, filter_inds[:, 1], nms_threshold)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    result = {}
    result['pred_boxes'] = boxes
    result['scores'] = scores
    result['pred_classes'] = filter_inds[:, 1]
    return result


def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_threshold: float,
        nms_threshold: float,
        topk_per_image: int,
) -> List[dict]:
    """Inference
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_threshold, nms_threshold, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return result_per_image


class FastRCNNDecoder(nn.Module):
    """Define decoder
    """
    def __init__(
            self,
            score_threshold: float = 0.05,
            nms_threshold: float = 0.5,
            topk_per_image=100
    ) -> None:
        super(FastRCNNDecoder, self).__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.topk_per_image = topk_per_image

    def forward(self, boxes, scores, image_shapes):
        """Forward
        """
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.score_threshold,
            self.nms_threshold,
            self.topk_per_image,
        )
