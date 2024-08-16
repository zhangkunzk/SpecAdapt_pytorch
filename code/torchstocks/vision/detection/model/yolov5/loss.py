#!/usr/bin/env python3

import math
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    'YoloLoss',
    'YoloLossAnchorFree'
]


def iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps: float = 1e-7):
    """compute iou
    """
    b1_x, b1_y, b1_w, b1_h = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2] * 0.5, bbox1[..., 3] * 0.5
    b2_x, b2_y, b2_w, b2_h = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2] * 0.5, bbox2[..., 3] * 0.5
    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h

    # Intersection area
    w_inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clip(0)
    h_inter = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clip(0)
    inter = w_inter * h_inter

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    return inter / union


def g_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps: float = 1e-7):
    """compute generalized iou
    """
    b1_x, b1_y, b1_w, b1_h = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2] * 0.5, bbox1[..., 3] * 0.5
    b2_x, b2_y, b2_w, b2_h = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2] * 0.5, bbox2[..., 3] * 0.5
    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h

    # Intersection area
    w_inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clip(0)
    h_inter = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clip(0)
    inter = w_inter * h_inter

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c_area = cw * ch + eps  # convex area
    return (inter / union) - (c_area - union) / c_area


def d_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps: float = 1e-7):
    """compute distance iou
    """
    b1_x, b1_y, b1_w, b1_h = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2] * 0.5, bbox1[..., 3] * 0.5
    b2_x, b2_y, b2_w, b2_h = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2] * 0.5, bbox2[..., 3] * 0.5
    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h

    # Intersection area
    w_inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clip(0)
    h_inter = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clip(0)
    inter = w_inter * h_inter

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    return (inter / union) - rho2 / c2


def c_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps: float = 1e-7):
    """compute complete iou
    """
    b1_x, b1_y, b1_w, b1_h = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2] * 0.5, bbox1[..., 3] * 0.5
    b2_x, b2_y, b2_w, b2_h = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2] * 0.5, bbox2[..., 3] * 0.5
    b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
    b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h

    # Intersection area
    w_inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clip(0)
    h_inter = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clip(0)
    inter = w_inter * h_inter

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    _iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - _iou + (1 + eps))
    return _iou - (rho2 / c2 + v * alpha)


class YoloLoss(nn.Module):
    """Yolo loss
    """

    def __init__(
            self,
            num_classes,
            anchors: List[torch.Tensor],
            strides: List[float],
            obj_weight: float,  # 0.1 * (3 / num_heads) * (image_size / 640) ** 2
            box_weight: float,  # 0.05 * (3 / num_heads)
            cls_weight: float,  # 0.5 * (3 / num_heads) * (num_classes / 80)
            anchor_t: float = 4.0  # 4.0
    ):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides
        self.anchor_t = anchor_t
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.obj_loss_balance = [4.0, 1.0, 0.4]

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        """Forward
        """
        with torch.no_grad():
            target_list = self.build_target(outputs, targets)

        device = outputs[0].device
        loss_box = torch.zeros((), dtype=torch.float32, device=device)
        loss_obj = torch.zeros((), dtype=torch.float32, device=device)
        loss_class = torch.zeros((), dtype=torch.float32, device=device)

        num_heads = len(self.anchors)
        assert num_heads == len(outputs)
        for i in range(num_heads):
            output = outputs[i]  # (n, a, h, w, d)
            image_idx, anchor_idx, grid_x_idx, grid_y_idx, box_true, class_true = target_list[i]

            # objectiveness (pred and true)
            obj_pred = output[..., 4]  # .sigmoid()
            obj_true = torch.zeros_like(obj_pred, device=device)

            if len(image_idx) != 0:
                # get subset from output based on target index
                subset = output[image_idx, anchor_idx, grid_y_idx, grid_x_idx]

                # box regression
                # xy = (subset[..., 0:2].sigmoid() * 2.0 - 0.5)
                # wh = (subset[..., 2:4].sigmoid() * 2.0).square() * (self.anchors[i][anchor_idx] / self.strides[i])
                # box_pred = torch.cat([xy, wh], -1)
                box_pred = subset[..., 0:4]
                box_iou = c_iou(box_pred, box_true)
                loss_box += (1.0 - box_iou).mean()

                # update obj_true
                obj_true[image_idx, anchor_idx, grid_y_idx, grid_x_idx] = box_iou.detach().clip(0, 1)

                # classification
                if self.num_classes > 1:  # multiple classes
                    prob_pred = subset[:, -self.num_classes:]  # .sigmoid()
                    prob_true = F.one_hot(class_true, self.num_classes).float()
                    loss_class += self.bce_cls(prob_pred, prob_true)

            # objectiveness (loss)
            loss_obj += self.bce_obj(obj_pred, obj_true)  # * self.obj_loss_balance[min(i, 2)]

        loss = self.obj_weight * loss_obj + self.box_weight * loss_box + self.cls_weight * loss_class
        loss = loss * outputs[0].shape[0]
        return loss, loss_obj, loss_box, loss_class

    def build_target(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        """Build target from bboxes for loss computing.

        Args:
            outputs: (n, a, h, w, d), float32
            targets: (nt, 6), float32, ixywhc

        Returns:
            image_idx, anchor_idx, grid_x_idx, grid_y_idx, box, clazz
        """
        targets = self._add_image_index(targets)  # xywhc to ixywhc
        targets = self._add_anchor_index(targets)  # ixywhc to aixywhc

        g = 0.5  # bias
        offsets = torch.tensor(
            [[0, 0], [g, 0], [0, g], [-g, 0], [0, -g]],
            dtype=torch.float32,
            device=targets.device
        )[:, None, :]  # (5, 1, 2)

        target_list = []
        num_heads = len(self.anchors)
        assert num_heads == len(outputs)
        for i in range(num_heads):
            output = outputs[i]  # (n, a, h, w, d)
            gh, gw = output.shape[2], output.shape[3]  # grid_height, grid_width
            gain1 = torch.tensor([gw, gh], dtype=torch.float32, device=self.anchors[i].device)
            anchors = self.anchors[i] / self.strides[i]  # (na, 2)
            gain2 = torch.tensor([1, 1, gw, gh, gw, gh, 1], dtype=torch.float32, device=targets.device)
            target_i = targets * gain2

            # "ratio" is the max ratio to anchor size. (ratio >= 1)
            # ratio = 1 means the box has (exactly) the same size with eth anchor.
            ratio = target_i[..., 4:6] / anchors[:, None, :]  # (na, nt, 2)
            ratio = torch.maximum(ratio, 1 / ratio).max(-1)[0]  # (na, nt)
            target_i = target_i[ratio < self.anchor_t]  # (?, 7), ? depends on how many targets matches the anchors

            # expand
            n = target_i.shape[0]
            left_top = target_i[:, 2:4]  # (?, 2)
            left_top_expand = (left_top % 1 < g) & (left_top > 1)  # (?, 2)
            left_expand, top_expand = left_top_expand[:, 0], left_top_expand[:, 1]  # (?,)
            right_bottom = gain1 - left_top  # (?, 2)
            right_bottom_expand = (right_bottom % 1 < g) & (right_bottom > 1)  # (?, 2)
            right_expand, bottom_expand = right_bottom_expand[:, 0], right_bottom_expand[:, 1]  # (?,)
            no_expand = torch.ones((n,), dtype=torch.bool, device=target_i.device)  # (?,)
            expand = torch.stack([no_expand, left_expand, top_expand, right_expand, bottom_expand])  # (5, ?)
            target_i = target_i.repeat((5, 1, 1))[expand]  # (?, 7)
            offset_i = offsets.repeat((1, n, 1))[expand]

            anchor_idx = target_i[:, 0].long()  # (?,)
            image_idx = target_i[:, 1].long()  # (?,)
            xy = target_i[:, 2:4]  # (?, 2)
            xy_grid = (xy - offset_i).floor()  # (?, 2)
            grid_x_idx = xy_grid[:, 0].long().clip(0, gw - 1)
            grid_y_idx = xy_grid[:, 1].long().clip(0, gh - 1)
            xy = xy - xy_grid  # (?, 2)
            wh = target_i[:, 4:6]  # (?, 2)
            box = torch.cat([xy, wh], -1)  # (?, 4)
            clazz = target_i[:, 6].long()  # (?,)
            target_list.append((image_idx, anchor_idx, grid_x_idx, grid_y_idx, box, clazz))

        return target_list

    @staticmethod
    def _add_image_index(targets: List[torch.Tensor]) -> torch.Tensor:
        bboxes_list = []
        for i, bboxes in enumerate(targets):
            assert len(bboxes.shape) == 2
            index_bboxes = torch.empty((len(bboxes), bboxes.shape[1] + 1), dtype=torch.float32, device=bboxes.device)
            index_bboxes[:, 0] = i
            index_bboxes[:, 1:] = bboxes
            bboxes_list.append(index_bboxes)
        targets = torch.cat(bboxes_list, 0)
        return targets

    def _add_anchor_index(self, targets: torch.Tensor) -> torch.Tensor:
        num_anchors = len(self.anchors[0])
        num_targets = len(targets)
        anchor_indices = torch.arange(num_anchors, device=targets.device, dtype=torch.float32)  # (na,)
        targets = [
            anchor_indices[:, None, None].repeat((1, num_targets, 1)),  # (na, nt, 1)
            targets[None, :, :].repeat((num_anchors, 1, 1))  # (na, nt, 6)
        ]
        targets = torch.cat(targets, -1)  # (na, nt, 7)
        return targets


class YoloLossAnchorFree(nn.Module):
    """Yolo loss for anchor-free method
    """

    def __init__(
            self,
            num_classes,
            strides: List[float],
            obj_weight: float,  # 0.1 * (3 / num_heads) * (image_size / 640) ** 2
            box_weight: float,  # 0.05 * (3 / num_heads)
            cls_weight: float,  # 0.5 * (3 / num_heads) * (num_classes / 80)
    ):
        super(YoloLossAnchorFree, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.obj_loss_balance = [4.0, 1.0, 0.4]

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        """Forward
        """
        with torch.no_grad():
            target_list = self.build_target(outputs, targets)
        device = outputs[0].device
        loss_box = torch.zeros((), dtype=torch.float32, device=device)
        loss_obj = torch.zeros((), dtype=torch.float32, device=device)
        loss_class = torch.zeros((), dtype=torch.float32, device=device)

        num_heads = len(outputs)
        for i in range(num_heads):
            output = outputs[i]  # (n, h, w, d)
            image_idx, grid_x_idx, grid_y_idx, box_true, class_true = target_list[i]

            # objectiveness (pred and true)
            obj_pred = output[..., 4]  # .sigmoid()
            obj_true = torch.zeros_like(obj_pred, device=device)

            if len(image_idx) != 0:
                # get subset from output based on target index
                subset = output[image_idx, grid_y_idx, grid_x_idx]
                # box regression
                box_pred = subset[..., 0:4]
                box_iou = c_iou(box_pred, box_true)
                loss_box += (1.0 - box_iou).mean()

                # update obj_true
                obj_true[image_idx, grid_y_idx, grid_x_idx] = box_iou.detach().clip(0, 1)

                # classification
                if self.num_classes > 1:  # multiple classes
                    prob_pred = subset[:, -self.num_classes:]  # .sigmoid()
                    prob_true = F.one_hot(class_true, self.num_classes).float()
                    loss_class += self.bce_cls(prob_pred, prob_true)

            # objectiveness (loss)
            loss_obj += self.bce_obj(obj_pred, obj_true)  # * self.obj_loss_balance[min(i, 2)]

        loss = self.obj_weight * loss_obj + self.box_weight * loss_box + self.cls_weight * loss_class
        loss = loss * outputs[0].shape[0]
        return loss, loss_obj, loss_box, loss_class

    def build_target(self, outputs: List[torch.Tensor], targets: List[torch.Tensor]):
        """Build target from bboxes for loss computing.

        Args:
            outputs: (n, h, w, d), float32
            targets: (nt, 6), float32, ixywhc

        Returns:
            image_idx, anchor_idx, grid_x_idx, grid_y_idx, box, clazz
        """
        targets = self._add_image_index(targets)  # xywhc to ixywhc

        g = 0.5  # bias
        offsets = torch.tensor(
            [[0, 0], [g, 0], [0, g], [-g, 0], [0, -g]],
            dtype=torch.float32,
            device=targets.device
        )[:, None, :]  # (5, 1, 2)

        target_list = []
        num_heads = len(outputs)
        for i in range(num_heads):
            output = outputs[i]  # (n, h, w, d)
            gh, gw = output.shape[1], output.shape[2]  # grid_height, grid_width
            gain1 = torch.tensor([gw, gh], dtype=torch.float32, device=targets.device)
            gain2 = torch.tensor([1, gw, gh, gw, gh, 1], dtype=torch.float32, device=targets.device)
            target_i = targets * gain2

            n = target_i.shape[0]
            left_top = target_i[:, 1:3]  # (?, 2)
            left_top_expand = (left_top % 1 < g) & (left_top > 1)  # (?, 2)
            left_expand, top_expand = left_top_expand[:, 0], left_top_expand[:, 1]  # (?,)
            right_bottom = gain1 - left_top  # (?, 2)
            right_bottom_expand = (right_bottom % 1 < g) & (right_bottom > 1)  # (?, 2)
            right_expand, bottom_expand = right_bottom_expand[:, 0], right_bottom_expand[:, 1]  # (?,)
            no_expand = torch.ones((n,), dtype=torch.bool, device=target_i.device)  # (?,)
            expand = torch.stack([no_expand, left_expand, top_expand, right_expand, bottom_expand])  # (5, ?)
            target_i = target_i.repeat((5, 1, 1))[expand]  # (?, 6)
            offset_i = offsets.repeat((1, n, 1))[expand]

            image_idx = target_i[:, 0].long()  # (?,)
            xy = target_i[:, 1:3]  # (?, 2)
            xy_grid = (xy - offset_i).floor()  # (?, 2)
            grid_x_idx = xy_grid[:, 0].long().clip(0, gw - 1)
            grid_y_idx = xy_grid[:, 1].long().clip(0, gh - 1)
            xy = xy - xy_grid  # (?, 2)
            wh = target_i[:, 3:5]  # (?, 2)
            box = torch.cat([xy, wh], -1)  # (?, 4)
            clazz = target_i[:, 5].long()  # (?,)
            target_list.append((image_idx, grid_x_idx, grid_y_idx, box, clazz))

        return target_list

    @staticmethod
    def _add_image_index(targets: List[torch.Tensor]) -> torch.Tensor:
        bboxes_list = []
        for i, bboxes in enumerate(targets):
            assert len(bboxes.shape) == 2
            index_bboxes = torch.empty((len(bboxes), bboxes.shape[1] + 1), dtype=torch.float32, device=bboxes.device)
            index_bboxes[:, 0] = i
            index_bboxes[:, 1:] = bboxes
            bboxes_list.append(index_bboxes)
        targets = torch.cat(bboxes_list, 0)
        return targets
