from typing import Dict, List, Optional, Tuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .poolers import ROIPooler
from .utils import Box2BoxTransform, Matcher, add_ground_truth_to_proposals, subsample_labels, pairwise_iou
from .losses import FastRCNNLoss


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    def __init__(
            self,
            input_size: int,
            num_classes: int = 20,
            cls_agnostic_bbox_reg: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = 4
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward
        """
        # x: per-region features of shape (N, ...) for N bounding boxes to predict.
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)  # (N, num_class+1)
        proposal_deltas = self.bbox_pred(x)  # (N, num_class*4) or (N, 4)
        return scores, proposal_deltas


class ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It typically contains logic to
    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads
    """

    def __init__(
            self,
            *,
            num_classes,
            batch_size_per_image,
            positive_fraction,
            proposal_matcher,
            proposal_append_gt=True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes  # number of foreground classes
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    def _sample_proposals(
            self,
            matched_idxs: torch.Tensor,
            matched_labels: torch.Tensor,
            gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of [0, 1]) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, bg_label=self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self,
            proposals: List[dict],
            targets: List[dict]
    ) -> List[dict]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than ``self.positive_fraction``.
        """
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image[:, :4], proposals_per_image['proposal_boxes']
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            del match_quality_matrix
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image[:, -1]
            )

            # Set target attributes of the sampled proposals:
            for k, v in proposals_per_image.items():
                proposals_per_image[k] = v[sampled_idxs]

            proposals_per_image['gt_classes'] = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # add gt_boxes
                proposals_per_image['gt_boxes'] = targets_per_image[:,
                                                  :4][sampled_targets]

            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_boxes fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward(self, x):
        """Forward
        """
        return x


class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(
            self,
            res5,
            res5_out_channels,
            in_features: List[str],
            pooler: ROIPooler,
            box2box_transform: Box2BoxTransform,
            fast_rcnn_losses: FastRCNNLoss,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5  # a CNN to compute per-region features
        self.in_features = in_features  # list of backbone feature map names
        self.pooler = pooler  # pooler to extra region features from backbone
        self.box2box_transform = box2box_transform
        box_predictor = FastRCNNOutputLayers(
            input_size=res5_out_channels,
            num_classes=self.num_classes
        )
        self.box_predictor = box_predictor  # make box predictions from the feature
        self.fast_rcnn_losses = fast_rcnn_losses

    def _shared_roi_transform(
            self,
            features: List[torch.Tensor],
            boxes: List[torch.Tensor]
    ):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def predict_boxes(
            self,
            predictions: Tuple[torch.Tensor, torch.Tensor],
            proposals: List[dict]
    ) -> Sequence:
        """Predict boxes
        """
        if len(proposals) == 0:
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p['proposal_boxes']) for p in proposals]
        proposal_boxes = torch.cat([p['proposal_boxes']
                                    for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes)  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self,
            predictions: Tuple[torch.Tensor, torch.Tensor],
            proposals: List[dict]
    ):
        """Predict
        """
        scores, _ = predictions
        num_inst_per_image = [len(p['objectness_logits']) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def inference(
            self,
            predictions: Tuple[torch.Tensor, torch.Tensor],
            proposals: List[dict]
    ):
        """Inference
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x['image_size'] for x in proposals]
        return boxes, scores, image_shapes

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[dict],
            targets: Optional[List[dict]] = None,
    ) -> Sequence:
        """Forward
        """
        if self.training:
            assert targets
            # proposals: dict_keys(['proposal_boxes', 'objectness_logits'])
            proposals = self.label_and_sample_proposals(proposals, targets)
            # proposals: dict_keys(['proposal_boxes', 'objectness_logits', 'gt_classes', 'gt_boxes'])
        del targets
        proposal_boxes = [x['proposal_boxes'] for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.fast_rcnn_losses(predictions, proposals)
            return [], losses
        else:
            pred_results = self.inference(predictions, proposals)
            return pred_results


def build_roi_heads(
        res5,
        res5_out_channels,
        in_features,
        pooler,
        num_classes,
        batch_size_per_image,
        box_reg_loss_type: str = "smooth_l1"
):
    """
    Build ROIHeads.
    """
    box2box_transform = Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0))
    fast_rcnn_losses = FastRCNNLoss(
        num_classes=num_classes,
        box2box_transform=box2box_transform,
        box_reg_loss_type=box_reg_loss_type,
    )

    return Res5ROIHeads(
        res5=res5,
        res5_out_channels=res5_out_channels,
        in_features=in_features,
        pooler=pooler,
        box2box_transform=box2box_transform,
        fast_rcnn_losses=fast_rcnn_losses,
        num_classes=num_classes,
        batch_size_per_image=batch_size_per_image,
        positive_fraction=0.25,
        proposal_matcher=Matcher([0.5], [0, 1], allow_low_quality_matches=False)
    )
