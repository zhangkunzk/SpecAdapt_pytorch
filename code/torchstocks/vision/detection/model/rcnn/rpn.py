from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torch import nn

from torchstocks.nn.vision import ConvBlock2d
from .utils import retry_if_cuda_oom
from .anchor_generator import build_anchor_generator
from .utils import Box2BoxTransform, Matcher, subsample_labels, pairwise_iou, boxes_clip, boxes_nonempty
from .losses import RPNLoss


class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(
            self,
            in_channels: int,
            num_anchors: int,
            box_dim: int = 4,
            conv_dims=None,
    ):
        super().__init__()
        if conv_dims is None:
            conv_dims = [-1]
        cur_channels = in_channels
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(
            cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

    def _get_rpn_conv(self, in_channels, out_channels):
        return ConvBlock2d(
            ch_in=in_channels,
            ch_out=out_channels,
            kernel=3,
            stride=1,
            padding=1
        )

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps
        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def __init__(
            self,
            in_features: List[str],
            head: nn.Module,
            rpn_losses: RPNLoss,
            anchor_generator: nn.Module,
            anchor_matcher: Matcher,
            box2box_transform: Box2BoxTransform,
            batch_size_per_image: int,
            positive_fraction: float,
            pre_nms_topk: Tuple[float, float],
            post_nms_topk: Tuple[float, float],
            nms_thresh: float = 0.7,
            min_box_size: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image  # 256
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: pre_nms_topk[0], False: pre_nms_topk[1]}  # 12000, 6000
        self.post_nms_topk = {
            True: post_nms_topk[0], False: post_nms_topk[1]}  # 2000, 1000
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.rpn_losses = rpn_losses

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, bg_label=0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        # scatter_() will be modified original Tensor, torch.Tensor.scatter_(dim, index, src) → Tensor
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(
            self,
            anchors: List[torch.Tensor],
            gt_instances: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Tensor]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            list[Tensor]:
                List of img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = torch.cat(anchors, dim=0)  # concat different feature map's anchors
        gt_boxes = [gt_instance[:, :4] for gt_instance in gt_instances]
        del gt_instances
        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            # gt_boxes_i: ground-truth boxes for i-th image  # torch.Tensor
            # Compute the iou in per-gt box and per-anchor box
            match_quality_matrix = retry_if_cuda_oom(
                pairwise_iou)(gt_boxes_i, anchors)  # shape=(num_gts, num_anchors)
            # get the gt index of max iou in anchor and gt, and label the anchor to 0, -1 or 1
            matched_idxs, gt_labels_i = retry_if_cuda_oom(
                self.anchor_matcher)(match_quality_matrix)
            # matched_idxs.shape = num_anchors(index in range [0, num_gts))
            # gt_lables_i.shape = num_anchors(0, -1, 1)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix
            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors)
            else:
                # TODO wasted indexing computation for ignored boxes
                # shape = (num_anchors, 4)
                matched_gt_boxes_i = gt_boxes_i[matched_idxs]
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            del matched_idxs, gt_labels_i
        return gt_labels, matched_gt_boxes

    def forward(
            self,
            image_sizes: List[Tuple[int, int]],
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[torch.Tensor]] = None,
    ):
        """Forward
        """
        box_dim = 4
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)  # [(Hi*Wi*15,4)]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)  # rearrange(score, 'n a h w -> n (h w a)')
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, box_dim, x.shape[-2], x.shape[-1]
                   ).permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(
                anchors, gt_instances)
            losses = self.rpn_losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes
        )
        return proposals, losses

    def predict_proposals(
            self,
            anchors: List[torch.Tensor],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """Predict proposals
        """
        with torch.no_grad():
            pred_proposals = self._decode_proposals(
                anchors, pred_anchor_deltas)  # [(N, Hi*Wi*A, B)]
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(
            self,
            anchors: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            N = pred_anchor_deltas_i.shape[0]  # N bounding boxes
            B = anchors_i.size(1)  # box_dim
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(
                0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


def find_top_rpn_proposals(
        proposals: List[torch.Tensor],
        pred_objectness_logits: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        nms_thresh: float,
        pre_nms_topk: int,
        post_nms_topk: int,
        min_box_size: float,
        training: bool
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.
    """
    num_images = len(image_sizes)
    device = proposals[0].device
    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        # proposals_i: (N, Hi*Wi*A, 4), logits_i: (N, Hi*Wi*A)
        Hi_Wi_A = logits_i.shape[1]
        # TODO: num_proposals_i = min(Hi_Wi_A//2, pre_nms_topk)
        num_proposals_i = min(Hi_Wi_A, pre_nms_topk)
        topk_scores_i, topk_idx = logits_i.topk(
            num_proposals_i, dim=1)  # each is (N, topk)
        # (N, topk, 4)
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,),
                                    level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        lvl = level_ids
        valid_mask = torch.isfinite(boxes).all(
            dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]

        boxes = boxes_clip(boxes, image_size)

        # filter empty boxes
        # min_box_size: minimum proposal box side length in pixels (absolute units wrt input images)
        keep = boxes_nonempty(boxes, threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = torchvision.ops.boxes.batched_nms(
            boxes.float(), scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]  # keep is already sorted
        res = {}
        res['proposal_boxes'] = boxes[keep]
        res['objectness_logits'] = scores_per_img[keep]
        results.append(res)
    return results


def build_rpn_head(input_shape):
    """
    Build an RPN head.
    """
    # Standard RPN is shared across levels:
    in_channels = [s.channels for s in input_shape]
    assert len(set(in_channels)) == 1, "Each level must have the same channel!"
    in_channels = in_channels[0]

    # RPNHead should take the same input as anchor generator
    anchor_generator = build_anchor_generator(input_shape)
    num_anchors = anchor_generator.num_anchors
    assert (
            len(set(num_anchors)) == 1
    ), "Each level must have the same number of anchors per spatial position"
    num_anchors = num_anchors[0]  # 15
    box_dim = 4
    conv_dims = [-1]
    return StandardRPNHead(in_channels, num_anchors, box_dim, conv_dims)


def build_proposal_generator(
        in_features,
        input_shape,
        rpn_box_reg_loss_type="smooth_l1",
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_pre_nms_topk_train=12000,  # per-image topk
        rpn_pre_nms_topk_test=6000,
        rpn_post_nms_topk_train=2000,
        rpn_post_nms_topk_test=1000,
        rpn_nms_thresh=0.7
):
    """
    Build a proposal generator.
    """
    loss_weight = {"loss_rpn_cls": 1.0, "loss_rpn_loc": 1.0 * 1.0}  # TODO: lambda=10
    box2box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    pre_nms_topk = (rpn_pre_nms_topk_train, rpn_pre_nms_topk_test)
    post_nms_topk = (rpn_post_nms_topk_train, rpn_post_nms_topk_test)
    head = build_rpn_head([input_shape[f] for f in in_features])

    anchor_generator = build_anchor_generator(
        [input_shape[f] for f in in_features]
    )

    anchor_matcher = Matcher(
        [0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True
    )

    rpn_losses = RPNLoss(
        batch_size_per_image=rpn_batch_size_per_image,
        box2box_transform=box2box_transform,
        box_reg_loss_type=rpn_box_reg_loss_type,
        loss_weight=loss_weight,
    )

    return RPN(
        in_features=in_features,
        head=head,
        rpn_losses=rpn_losses,
        anchor_generator=anchor_generator,
        anchor_matcher=anchor_matcher,
        box2box_transform=box2box_transform,
        batch_size_per_image=rpn_batch_size_per_image,
        positive_fraction=rpn_positive_fraction,
        pre_nms_topk=pre_nms_topk,
        post_nms_topk=post_nms_topk,
        nms_thresh=rpn_nms_thresh,
        min_box_size=0
    )
