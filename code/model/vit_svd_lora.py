#!/usr/bin/env python3


import math
import torch
import timm
from torch import nn
from timm.models import create_model

__all__ = [
    'Model'
]

@torch.no_grad()
def adapt_weight(inner_p, set_rank, largest_singulars: bool = False):
    u0, s, vh0 = torch.linalg.svd(inner_p)
    if largest_singulars:
        u0 = u0[..., :set_rank]
        s = s[..., None, :set_rank]
        vh0 = vh0[..., :set_rank, :]
    else:
        rank = s.shape[-1]
        u0 = u0[..., rank - set_rank:rank]
        s = s[..., None, -set_rank:]
        vh0 = vh0[..., rank - set_rank:rank, :]
    return u0, s, vh0

def slora_forward_attn(module, x):
    B, N, C = x.shape
    qkv = module.qkv(x)
    q = module.dp(module.q_slora_b(module.q_s * module.q_slora_a(x)))
    k = module.dp(module.k_slora_b(module.k_s * module.k_slora_a(x)))
    v = module.dp(module.v_slora_b(module.v_s * module.v_slora_a(x)))
    qkv[:, :, :C] += q
    qkv[:, :, C:-C] += k
    qkv[:, :, -C:] += v

    qkv = qkv.reshape(B, N, 3,
                      module.num_heads,
                      C // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)
    attn = module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = module.proj(x)
    proj += module.dp(module.proj_slora_b(module.proj_s * module.proj_slora_a(x)))
    x = module.proj_drop(proj)
    return x


def slora_forward_mlp(module, x):
    h = module.fc1(x)
    h += module.dp(module.fc1_slora_b(module.fc1_s * module.fc1_slora_a(x)))
    x = module.act(h)
    x = module.drop1(x)
    h = module.fc2(x)
    h += module.dp(module.fc2_slora_b(module.fc2_s * module.fc2_slora_a(x)))
    x = module.drop2(h)
    return x


def set_slora(model: nn.Module, embed_size=768, rank=8, reverse_s=False):
    for _ in model.children():
        if isinstance(_, timm.models.vision_transformer.Attention):
            origin_weight_data = _.qkv.weight.data

            origin_weight_data_q =  origin_weight_data[:embed_size, :]
            u0, s, vh0 = adapt_weight(origin_weight_data_q, set_rank=rank)
            _.qkv.weight.data[:embed_size, :] = origin_weight_data_q - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.q_slora_a = nn.Linear(embed_size, rank, bias=False)
            _.q_s = nn.Parameter(s, requires_grad=False)
            _.q_slora_b = nn.Linear(rank, embed_size, bias=False)

            origin_weight_data_k = origin_weight_data[embed_size:-embed_size, :]
            u0, s, vh0 = adapt_weight(origin_weight_data_k, set_rank=rank)
            _.qkv.weight.data[embed_size:-embed_size, :] = origin_weight_data_k - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.k_slora_a = nn.Linear(embed_size, rank, bias=False)
            _.k_s = nn.Parameter(s, requires_grad=False)
            _.k_slora_b = nn.Linear(rank, embed_size, bias=False)

            origin_weight_data_v = origin_weight_data[-embed_size:, :]
            u0, s, vh0 = adapt_weight(origin_weight_data_v, set_rank=rank)
            _.qkv.weight.data[-embed_size:, :] = origin_weight_data_v - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.v_slora_a = nn.Linear(embed_size, rank, bias=False)
            _.v_s = nn.Parameter(s, requires_grad=False)
            _.v_slora_b = nn.Linear(rank, embed_size, bias=False)

            origin_weight_data = _.proj.weight.data
            u0, s, vh0 = adapt_weight(origin_weight_data, set_rank=rank)
            _.proj.weight.data = origin_weight_data - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.proj_slora_a = nn.Linear(embed_size, rank, bias=False)
            _.proj_s = nn.Parameter(s, requires_grad=False)
            _.proj_slora_b = nn.Linear(rank, embed_size, bias=False)

            _.dp = nn.Dropout(0.1)
            bound_method = slora_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)

        elif isinstance(_, timm.models.layers.mlp.Mlp):
            origin_weight_data = _.fc1.weight.data
            u0, s, vh0 = adapt_weight(origin_weight_data, set_rank=rank)
            _.fc1.weight.data = origin_weight_data - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.fc1_slora_a = nn.Linear(embed_size, rank, bias=False)
            _.fc1_s = nn.Parameter(s, requires_grad=False)
            _.fc1_slora_b = nn.Linear(rank, embed_size * 4, bias=False)

            origin_weight_data = _.fc2.weight.data
            u0, s, vh0 = adapt_weight(origin_weight_data, set_rank=rank)
            _.fc2.weight.data = origin_weight_data - (u0 * s) @ vh0
            if reverse_s:
                s = torch.flip(s, dims=[-1])  # reverse s
            _.fc2_slora_a = nn.Linear(embed_size * 4, rank, bias=False)
            _.fc2_s = nn.Parameter(s, requires_grad=False)
            _.fc2_slora_b = nn.Linear(rank, embed_size, bias=False)

            _.dp = nn.Dropout(0.1)
            bound_method = slora_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_slora(_, embed_size=768, rank=rank)


class Model(nn.Module):

    def __init__(
            self,
            num_classes: int = None,
            backbone_name: str = 'vit_base_patch16_224',
            drop_rate: float = 0.0,
            rank: int = 4,
            freeze_backbone: bool = True
    ) -> None:
        super(Model, self).__init__()
        self.model = Network(
            num_classes=num_classes,
            backbone_name=backbone_name,
            drop_rate=drop_rate
        )
        set_slora(model=self.model, rank=rank)
        self.loss = nn.CrossEntropyLoss()
        self.freeze_backbone = freeze_backbone
        self.init_new_weight(self.model)

    def init_new_weight(self, model: nn.Module):
        for n, p in model.named_parameters():
            if 'slora_a' in n:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif 'slora_b' in n:
                nn.init.zeros_(p)
            else:
                if self.freeze_backbone and (not 'head' in n):
                    p.requires_grad=False


    def forward(self, inputs, targets=None):
        y_ = self.model(inputs)
        if targets is None:
            return y_.argmax(-1)
        else:
            return self.loss(y_, targets)


class Network(nn.Module):

    def __init__(
            self,
            num_classes: int,
            backbone_name: str = 'vit_base_patch16_224',
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.1
    ) -> None:
        super(Network, self).__init__()
        self.backbone = create_model(
            backbone_name,
            checkpoint_path='output/ViT-B_16.npz',
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = Model(num_classes=100, rank=8)
    x = torch.rand((2, 3, 224, 224))
    total = 0
    result = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'head' not in n:
                total += p.numel()
    print(total / 1e6)