#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$1 python train.py \
  --batch_size=64 \
  --model 'model.vit.Model' \
  --num_epochs=100 \
  --dataset_name="$2" \
  --optimizer 'LoRAAdamW' \
  --rank $3 \
  --weight_decay 0.2 \
  --drop_rate 0.0 \
  --param_groups '[{"name": "model.backbone", "lr": 0}, {"match": ["^.*(blocks).*(attn.(qkv|proj).*weight)$", "^.*mlp.*weight$", "^.*(blocks).*(attn.(qkv|proj).*bias)$", "^.*mlp.*bias$"], "lr": 1e-3, "tag": "low_rank"}, {"name": "model.backbone.head", "lr": 1e-3}]'
