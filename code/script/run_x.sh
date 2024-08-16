#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$1 python train.py \
  --batch_size=64 \
  --model 'model.vit.Model' \
  --num_epochs=100 \
  --dataset_name="$2" \
  --optimizer 'SAdamW' \
  --rank $3 \
  --max_lr 3e-3 \
  --lr_decay_min_value 0.0 \
  --momentum 0.9 \
  --weight_decay 0.3 \
  --drop_rate 0.0 \
  --drop_path_rate 0.1 \
  --param_groups '[
    {"name": "model.backbone", "lr": 0},
    {
      "match": [
        "^.*weight$",
        "^.*bias$"
      ],
      "tag": "low_rank"
    },
    {"name": "model.backbone.head"}
  ]'
#  --param_groups '[
#    {"name": "model.backbone", "lr": 0},
#    {
#      "match": [
#        "^.*patch_embed.proj.weight$",
#        "^.*(blocks).*(attn.(qkv|proj).*weight)$",
#        "^.*mlp.*weight$"
#      ],
#      "tag": "low_rank"
#    },
#    {"name": "model.backbone.head"}
#  ]'
