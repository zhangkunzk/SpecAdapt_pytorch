# Image Classification

## 基于imagenet预训练权重进行训练
For training on single GPU:
```bash
python -m torchstocks.vision.classification.train \
    --model torchstocks.vision.classification.model.resnet.Model \
    --backbone torchstocks.models.imagenet.resnet34 \
    --pretrained True \
    --param_groups '[{"name": "model.backbone", "lr": 1e-5, "mode":False}]' \
    --num_classes ${classe number} \
    --train_data_path ${path} \
    --test_data_path ${path} \
    --train_file ${path} \
    --test_file ${path} \
    --image_size 224 \
    --p_flip_lr 0.5 \
    --p_resize 0.5 \
    --rnd_resize 0.5 \
    --p_color 0.2 \
    --rnd_hue 0.05 \
    --rnd_saturation 0.2 \
    --rnd_brightness 0.2 \
    --rnd_contrast 0.3 \
    --p_rotate 0.2 \
    --rnd_rotate 10 \
    --batch_size 256 \
    --optimizer AdamW \
    --momentum 0.9 \
    --max_lr 1e-3 \
    --weight_decay 0.3 \
    --clip_grad_norm 0.1 \
    --num_epochs 100
```

## Train on ImageNet Dataset (ILSVRC2012)

For training on single GPU:

```bash
python -m torchstocks.vision.classification.train \
    --model torchstocks.vision.classification.model.resnet.Model \
    --backbone torchstocks.models.imagenet.resnet34 \
    --num_classes 1000 \
    --train_data_path /mnt/cephfs/data/ilsvrc2012/ \
    --test_data_path /mnt/cephfs/data/ilsvrc2012/ \
    --train_file train-256-98.ds \
    --test_file valid-256-98.ds \
    --image_size 224 \
    --p_flip_lr 0.5 \
    --p_resize 0.5 \
    --rnd_resize 0.5 \
    --p_color 0.2 \
    --rnd_hue 0.05 \
    --rnd_saturation 0.2 \
    --rnd_brightness 0.2 \
    --rnd_contrast 0.3 \
    --p_rotate 0.2 \
    --rnd_rotate 10 \
    --batch_size 256 \
    --optimizer AdamW \
    --momentum 0.9 \
    --max_lr 1e-3 \
    --weight_decay 0.3 \
    --clip_grad_norm 0.1 \
    --num_epochs 100
```

For training on multi-GPUs, e.g., GPU #0, #1, #2, #3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node auto -m torchstocks.vision.classification.train \
    --model torchstocks.vision.classification.model.resnet.Model \
    --backbone torchstocks.models.imagenet.resnet34 \
    --num_classes 1000 \
    --train_data_path /mnt/cephfs/data/ilsvrc2012/ \
    --test_data_path /mnt/cephfs/data/ilsvrc2012/ \
    --train_file train-256-98.ds \
    --test_file valid-256-98.ds \
    --image_size 224 \
    --p_flip_lr 0.5 \
    --p_resize 0.5 \
    --rnd_resize 0.5 \
    --p_color 0.2 \
    --rnd_hue 0.05 \
    --rnd_saturation 0.2 \
    --rnd_brightness 0.2 \
    --rnd_contrast 0.3 \
    --p_rotate 0.2 \
    --rnd_rotate 10 \
    --batch_size 200 \
    --optimizer AdamW \
    --momentum 0.9 \
    --max_lr 1e-3 \
    --weight_decay 0.3 \
    --clip_grad_norm 0.1 \
    --num_epochs 100
```

```
==== Model config ====
   backbone: 'torchstocks.models.imagenet.resnet34'
num_classes: 1000
    non_lin: None
       norm: None
   pretrain: False
==== Augmenter config ====
     p_flip_lr: 0.5
       p_color: 0.2
       rnd_hue: 0.05
rnd_saturation: 0.2
rnd_brightness: 0.2
  rnd_contrast: 0.3
      p_rotate: 0.2
    rnd_rotate: 10
      p_resize: 0.5
    rnd_resize: 0.5
==== Train Dataset config ====
        path: '/mnt/cephfs/data/ilsvrc2012/train-256-98.ds'
  image_size: 224
shorter_side: None
 longer_side: None
   augmenter: BasicImageAugmenter(name=UnnamedBasicImageAugmenter, random_orde...
==== Trainer config ====
         model: Model instance at 22736440137120
 train_dataset: TrainDataset instance at 22736416838208
  test_dataset: TestDataset instance at 22736416838112
     optimizer: 'AdamW'
    batch_size: 200
        max_lr: 0.001
      momentum: 0.9
  weight_decay: 0.3
    num_epochs: 100
   num_workers: 10
  param_groups: None
clip_grad_norm: 0.1
        device: 'cuda'
[1/100] L=5.194612 Acc=0.0705 F1=0.0512
[2/100] L=4.122496 Acc=0.1462 F1=0.1270
[3/100] L=3.471670 Acc=0.2107 F1=0.2013
[4/100] L=2.963911 Acc=0.3360 F1=0.3202
[5/100] L=2.677873 Acc=0.3355 F1=0.3294
[6/100] L=2.548738 Acc=0.3777 F1=0.3729
[7/100] L=2.469043 Acc=0.4285 F1=0.4214
[8/100] L=2.236700 Acc=0.4484 F1=0.4461
[9/100] L=2.195549 Acc=0.4576 F1=0.4497
[10/100] L=2.231694 Acc=0.4476 F1=0.4436
[11/100] L=2.186497 Acc=0.4497 F1=0.4441
[12/100] L=2.164139 Acc=0.4698 F1=0.4638
[13/100] L=2.182333 Acc=0.4638 F1=0.4594
[14/100] L=2.094678 Acc=0.4472 F1=0.4394
[15/100] L=2.216921 Acc=0.4561 F1=0.4569
[16/100] L=2.209790 Acc=0.4434 F1=0.4373
[17/100] L=2.081695 Acc=0.4159 F1=0.4179
[18/100] L=2.100718 Acc=0.4644 F1=0.4599
[19/100] L=2.105096 Acc=0.4790 F1=0.4770
[20/100] L=2.000555 Acc=0.4665 F1=0.4632
[21/100] L=2.007523 Acc=0.5096 F1=0.5068
[22/100] L=2.040178 Acc=0.4656 F1=0.4591
[23/100] L=2.053765 Acc=0.4813 F1=0.4805
[24/100] L=2.009422 Acc=0.4703 F1=0.4678
[25/100] L=2.038072 Acc=0.4859 F1=0.4821
[26/100] L=1.992676 Acc=0.4603 F1=0.4574
[27/100] L=1.899436 Acc=0.5140 F1=0.5076
[28/100] L=2.011234 Acc=0.4982 F1=0.4927
[29/100] L=1.919462 Acc=0.5075 F1=0.5034
[30/100] L=1.945873 Acc=0.5082 F1=0.5039
[31/100] L=1.902770 Acc=0.5198 F1=0.5163
[32/100] L=1.954590 Acc=0.5048 F1=0.5004
[33/100] L=1.892125 Acc=0.5203 F1=0.5156
[34/100] L=1.917568 Acc=0.5256 F1=0.5228
[35/100] L=1.862272 Acc=0.4971 F1=0.4933
[36/100] L=1.849631 Acc=0.5074 F1=0.5049
[37/100] L=1.774643 Acc=0.5223 F1=0.5187
[38/100] L=1.889957 Acc=0.5161 F1=0.5113
[39/100] L=1.840074 Acc=0.5445 F1=0.5397
[40/100] L=1.776226 Acc=0.4995 F1=0.4984
[41/100] L=1.758668 Acc=0.5461 F1=0.5430
[42/100] L=1.747182 Acc=0.5145 F1=0.5118
[43/100] L=1.801028 Acc=0.5383 F1=0.5340
[44/100] L=1.840123 Acc=0.5162 F1=0.5162
[45/100] L=1.785288 Acc=0.5296 F1=0.5235
[46/100] L=1.790769 Acc=0.5604 F1=0.5574
[47/100] L=1.715631 Acc=0.5653 F1=0.5603
[48/100] L=1.796026 Acc=0.5475 F1=0.5453
[49/100] L=1.702496 Acc=0.5595 F1=0.5549
[50/100] L=1.702502 Acc=0.5584 F1=0.5562
[51/100] L=1.669651 Acc=0.5752 F1=0.5717
[52/100] L=1.655216 Acc=0.5355 F1=0.5345
[53/100] L=1.675625 Acc=0.5773 F1=0.5719
[54/100] L=1.631431 Acc=0.5589 F1=0.5562
[55/100] L=1.563123 Acc=0.5885 F1=0.5845
[56/100] L=1.618544 Acc=0.5916 F1=0.5901
[57/100] L=1.570500 Acc=0.5895 F1=0.5859
[58/100] L=1.517292 Acc=0.6087 F1=0.6049
[59/100] L=1.570569 Acc=0.6013 F1=0.5979
[60/100] L=1.509426 Acc=0.5920 F1=0.5889
[61/100] L=1.485593 Acc=0.6023 F1=0.6001
[62/100] L=1.493330 Acc=0.6173 F1=0.6149
[63/100] L=1.475765 Acc=0.6199 F1=0.6163
[64/100] L=1.427696 Acc=0.6190 F1=0.6148
[65/100] L=1.456342 Acc=0.6130 F1=0.6114
[66/100] L=1.402184 Acc=0.6297 F1=0.6248
[67/100] L=1.361074 Acc=0.6305 F1=0.6276
[68/100] L=1.323637 Acc=0.6342 F1=0.6297
[69/100] L=1.362991 Acc=0.6347 F1=0.6308
[70/100] L=1.335752 Acc=0.6290 F1=0.6263
[71/100] L=1.308692 Acc=0.6598 F1=0.6564
[72/100] L=1.306508 Acc=0.6572 F1=0.6536
[73/100] L=1.235194 Acc=0.6532 F1=0.6495
[74/100] L=1.237727 Acc=0.6617 F1=0.6576
[75/100] L=1.198767 Acc=0.6672 F1=0.6640
[76/100] L=1.145895 Acc=0.6762 F1=0.6723
[77/100] L=1.165360 Acc=0.6799 F1=0.6761
[78/100] L=1.151040 Acc=0.6805 F1=0.6773
[79/100] L=1.122379 Acc=0.6868 F1=0.6835
[80/100] L=1.105157 Acc=0.6933 F1=0.6899
[81/100] L=1.001722 Acc=0.6841 F1=0.6814
[82/100] L=1.071142 Acc=0.6971 F1=0.6940
[83/100] L=0.911463 Acc=0.6973 F1=0.6939
[84/100] L=0.943055 Acc=0.7073 F1=0.7035
[85/100] L=0.914314 Acc=0.7074 F1=0.7044
[86/100] L=0.933216 Acc=0.7115 F1=0.7088
[87/100] L=0.890557 Acc=0.7173 F1=0.7140
[88/100] L=0.892693 Acc=0.7180 F1=0.7150
[89/100] L=0.831040 Acc=0.7207 F1=0.7178
[90/100] L=0.842491 Acc=0.7259 F1=0.7229
[91/100] L=0.786993 Acc=0.7289 F1=0.7260
[92/100] L=0.707492 Acc=0.7302 F1=0.7272
[93/100] L=0.715386 Acc=0.7326 F1=0.7299
[94/100] L=0.704283 Acc=0.7339 F1=0.7311
[95/100] L=0.732889 Acc=0.7348 F1=0.7319
[96/100] L=0.690613 Acc=0.7350 F1=0.7320
[97/100] L=0.700061 Acc=0.7358 F1=0.7332
[98/100] L=0.667650 Acc=0.7357 F1=0.7330
[99/100] L=0.703018 Acc=0.7366 F1=0.7338
[100/100] L=0.694683 Acc=0.7367 F1=0.7340
```

