# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional, Tuple, Union, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchstocks.common.entry import load_state_dict
from torchstocks.nn.normalization import LayerNorm2d

DefaultNorm = partial(nn.LayerNorm, eps=1e-6)
DefaultAct = nn.GELU
WEIGHT_DICT = {
    'vit_base': '/mnt/cephfs/data/pretrained_weight/sam/sam_vit_b_encoder.npz',
    'vit_large': '/mnt/cephfs/data/pretrained_weight/sam/sam_vit_l_encoder.npz',
    'vit_huge': '/mnt/cephfs/data/pretrained_weight/sam/sam_vit_h_encoder.npz'
}


class ImageEncoderViT(nn.Module):

    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 1024,
            patch_size: int = 16,
            in_channels: int = 3,
            emb_size: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_channels: int = 256,
            qkv_bias: bool = True,
            norm_type: Callable[[int], nn.Module] = DefaultNorm,
            act_type: Callable[[], nn.Module] = DefaultAct,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """ImageEncoderViT
        This class and its supporting functions below lightly adapted from the ViTDet backbone available at:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa

        Args:
            image_size (int, tuple): Input image size.
            patch_size (int): Patch size.
            in_channels (int): Number of input image channels.
            emb_size (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_type (Callable[[int], nn.Module]): Normalization layer.
            act_type (Callable[[], nn.Module]): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = (image_size, image_size) if isinstance(image_size, int) else image_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_channels=in_channels,
            emb_size=emb_size,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.img_size[0] // patch_size, self.img_size[1] // patch_size, emb_size)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                feature_size=emb_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_type=norm_type,
                act_type=act_type,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(self.img_size[0] // patch_size, self.img_size[1] // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                emb_size,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_size = emb_size
        self.global_attn_indexes = global_attn_indexes
        self.ch_out_list = [emb_size] * 4
        self.ch_out_list.append(out_channels)
        self.stride_list = [patch_size] * 5

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """output feature list"""
        output = []
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed
            if x.shape[1] != pos_embed.shape[1] or x.shape[2] != pos_embed.shape[2]:
                pos_embed = F.interpolate(
                    self.pos_embed.permute(0, 3, 1, 2),
                    size=(x.shape[1], x.shape[2]),
                    mode='bicubic',
                    align_corners=False,
                ).permute((0, 2, 3, 1))
            x = x + pos_embed

        for index, blk in enumerate(self.blocks):
            x = blk(x)
            if index in self.global_attn_indexes:
                output.append(x.permute(0, 3, 1, 2))

        x = self.neck(x.permute(0, 3, 1, 2))
        output.append(x)
        return output


class PatchEmbed(nn.Module):

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_channels: int = 3,
            emb_size: int = 768,
    ) -> None:
        """Image to Patch Embedding

        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_channels (int): Number of input image channels.
            emb_size (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Block(nn.Module):

    def __init__(
            self,
            feature_size: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_type: Callable[[int], nn.Module] = DefaultNorm,
            act_type: Callable[[], nn.Module] = DefaultAct,
            use_rel_pos: bool = False,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Transformer blocks with support of window attention

        Args:
            feature_size (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_type (Callable[[int], nn.Module]): Normalization layer.
            act_type (Callable[[], nn.Module]): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_type(feature_size)
        self.attn = Attention(
            feature_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_type(feature_size)
        self.mlp = MLPBlock(emb_size=feature_size, mlp_size=int(feature_size * mlp_ratio), act_type=act_type)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            # Partition -> Attention -> Reverse partition
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = Block.window_partition(x, self.window_size)
            x = self.attn(x)
            x = Block.window_unpartition(x, self.window_size, pad_hw, (h, w))
        else:
            # Global attention
            x = self.attn(x)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    @staticmethod
    def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    @staticmethod
    def window_unpartition(
            windows: torch.Tensor,
            window_size: int,
            pad_hw: Tuple[int, int],
            hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
            window_size (int): window size.
            pad_hw (Tuple): padded height and width (Hp, Wp).
            hw (Tuple): original height and width (H, W) before padding.

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            feature_size: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            feature_size (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            input_size (tuple[int, int]|None): Input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = feature_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(feature_size, feature_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(feature_size, feature_size)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = Attention.add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

    @staticmethod
    def add_decomposed_rel_pos(
            attn: torch.Tensor,
            q: torch.Tensor,
            rel_pos_h: torch.Tensor,
            rel_pos_w: torch.Tensor,
            q_size: Tuple[int, int],
            k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = Attention.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = Attention.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
                attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    @staticmethod
    def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.permute(1, 0).unsqueeze(0),
                size=max_rel_dist,
                mode='linear',
            )
            rel_pos_resized = rel_pos_resized.squeeze(0).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]


class MLPBlock(nn.Module):
    """mlp block without dropout"""

    def __init__(
            self,
            emb_size: int,
            mlp_size: int,
            act_type: Callable[[], nn.Module] = DefaultAct,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(emb_size, mlp_size)
        self.lin2 = nn.Linear(mlp_size, emb_size)
        self.act = act_type()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def vit_base(image_size: int = 1024, pretrained: bool = True):
    """vit base
    """
    vit = ImageEncoderViT(
        image_size=image_size,
        patch_size=16,
        emb_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_channels=256,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11)
    )
    if pretrained:
        load_state_dict(vit, WEIGHT_DICT['vit_base'])
    return vit


def vit_large(image_size: int = 1024, pretrained: bool = True):
    """vit large
    """
    vit = ImageEncoderViT(
        image_size=image_size,
        patch_size=16,
        emb_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        out_channels=256,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=(5, 11, 17, 23)
    )
    if pretrained:
        load_state_dict(vit, WEIGHT_DICT['vit_base'])
    return vit


def vit_huge(image_size: int = 1024, pretrained: bool = True):
    """vit huge
    """
    vit = ImageEncoderViT(
        image_size=image_size,
        patch_size=16,
        emb_size=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        out_channels=256,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=(7, 15, 23, 31)
    )
    if pretrained:
        load_state_dict(vit, WEIGHT_DICT['vit_base'])
    return vit


################################################################################
# Code for debug  and test.
################################################################################

def test():
    x = torch.rand((2, 3, 512, 512))
    model = vit_base()
    feature_list = model(x)

    for feature in feature_list:
        print(feature.shape)
    print(model.ch_out_list)
    print(model.stride_list)
    return 0


if __name__ == '__main__':
    exit(test())
