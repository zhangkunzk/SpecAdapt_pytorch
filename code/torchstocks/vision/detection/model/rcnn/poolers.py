from typing import List

import torch
from torch import nn
from torchvision.ops import roi_align
from torchvision import __version__

__all__ = ["ROIPooler"]


def convert_boxes_to_pooler_format(box_lists: List[torch.Tensor]):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops.
    Returns:
        When input is list[tensor]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
    """
    boxes = torch.cat(box_lists, dim=0)
    sizes = torch.as_tensor([x.__len__()
                             for x in box_lists], device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(box_lists), dtype=boxes.dtype,
                     device=boxes.device), sizes
    )
    return torch.cat([indices[:, None], boxes], dim=1)


# NOTE: torchvision's RoIAlign has a different default aligned=False
class ROIAlign(nn.Module):
    """Define roi align
    """
    def __init__(
            self,
            output_size: tuple,
            spatial_scale: float,
            sampling_ratio: int,
            aligned: bool = True
    ) -> None:
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized:
            input = input.dequantize()
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales
    ) -> None:
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(
            output_size[1], int)
        self.output_size = output_size

        self.level_poolers = nn.ModuleList(
            ROIAlign(
                output_size, spatial_scale=scale, sampling_ratio=0, aligned=True
            )
            for scale in scales
        )

    def forward(self, x: List[torch.Tensor], box_lists: List[torch.Tensor]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Tensor]):
                A list of N tensor , where N is the number of images in the batch.
        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), f'unequal value, num_level_assignments={num_level_assignments}, but x is list of {len(x)} Tensors'
        assert len(box_lists) == x[0].size(
            0
        ), f'unequal value, x[0] batch dim 0 is {x[0].size(0)}, but box_list has length {len(box_lists)}'
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype
            )  # (0, 256, 14, 14)

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        assert num_level_assignments == 1, "only support level poolers is 1"
        output = self.level_poolers[0](x[0], pooler_fmt_boxes)
        return output
