#!/usr/bin/env python3


from typing import List

import numpy as np
import torch
import torchvision
from torch import nn

__all__ = [
    'non_max_suppression',
    'YoloDecoder'
]


def xywh_to_xyxy(xywh):
    """cxcywh data format to xyxy format
    """
    xyxy = xywh.clone() if isinstance(xywh, torch.Tensor) else np.copy(xywh)
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    w = w * 0.5
    h = h * 0.5
    xyxy[..., 0] = x - w
    xyxy[..., 1] = y - h
    xyxy[..., 2] = x + w
    xyxy[..., 3] = y + h
    return xyxy


def xyxy_to_xywh(xyxy):
    """xyxy data format to cxcywh format
    """
    xywh = xyxy.clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy)
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    xywh[..., 0] = (x1 + x2) * 0.5
    xywh[..., 1] = (y1 + y2) * 0.5
    xywh[..., 2] = x2 - x1
    xywh[..., 3] = y2 - y1
    return xywh


def non_max_suppression(
        prediction: torch.Tensor,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.6,
        max_detections: int = 300,
        multi_label: bool = True
) -> List[torch.Tensor]:
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction:
        conf_threshold:
        iou_threshold:
        max_detections:
        multi_label:

    Returns:
        list of detections, a (?, 6) tensor per image [xywh, cls, conf]
    """
    num_class = prediction.shape[2] - 5  # number of classes
    candidates = prediction[..., 4] > conf_threshold  # candidates

    assert 0 <= conf_threshold <= 1, f'Invalid conf_threshold {conf_threshold}. It should be in [0.0, 1.0].'
    assert 0 <= iou_threshold <= 1, f'Invalid iou_threshold {iou_threshold}. It should be in [0.0, 1.0].'

    # Settings
    _, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= num_class > 1  # multiple labels per box (adds 0.5ms/img)
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for index, pred_output in enumerate(prediction):  # image index, image inference
        # Apply constraints
        pred_output = pred_output[candidates[index]]  # confidence

        # If none remain process next image
        if not pred_output.shape[0]:
            continue

        # Compute conf
        pred_output[:, 5:] *= pred_output[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh_to_xyxy(pred_output[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (pred_output[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
            pred_output = torch.cat((box[i], pred_output[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = pred_output[:, 5:].max(1, keepdim=True)
            pred_output = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # Check shape
        num_boxes = pred_output.shape[0]  # number of boxes
        if not num_boxes:  # no boxes
            continue
        elif num_boxes > max_nms:  # excess boxes
            pred_output = pred_output[pred_output[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = pred_output[:, 5:6] * max_wh  # classes
        boxes, scores = pred_output[:, :4] + c, pred_output[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_detections:  # limit detections
            i = i[:max_detections]

        output[index] = torch.cat([
            xyxy_to_xywh(pred_output[i][:, :4]),
            pred_output[i][:, -1].unsqueeze(-1),
            pred_output[i][:, -2].unsqueeze(-1)
        ], 1)  # xywh,class,confidence

    return output


class YoloDecoder(nn.Module):
    """Yolo decoder
    """

    def __init__(self,
                 obj_threshold: float = 0.001,
                 nms_threshold: float = 0.6,
                 max_num_bboxes=300,
                 multi_label=True):
        super(YoloDecoder, self).__init__()
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.max_num_bboxes = max_num_bboxes
        self.multi_label = multi_label

    def forward(self, flat_outputs: torch.Tensor):
        """Forward
        """
        if flat_outputs.shape[1] == 0:
            return []

        last_output = non_max_suppression(
            flat_outputs,
            conf_threshold=self.obj_threshold,
            iou_threshold=self.nms_threshold,
            max_detections=self.max_num_bboxes,
            multi_label=self.multi_label
        )
        return last_output
