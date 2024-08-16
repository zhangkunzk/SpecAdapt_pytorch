# 对比学习

## 参数

略。


## 示例
单卡，model包含backbone和分类层
```bash
CUDA_VISIBLE_DEVICES=0 python -m torchstocks.vision.contrastive.train \
--network torchstocks.models.imagenet.resnet18 --num_classes 512 \
--data_path ~/data/stl10/ --image_size 96 --image_field feature \
--emb_size 512 --head_size 128
```
多卡ddp，model仅包含backbone无分类层
```bash
CUDA_VISIBLE_DEVICES='3, 4, 5, 6' \
torchrun --nproc_per_node=4 --master_port 29800 -m  torchstocks.vision.contrastive.train \
--network torchstocks.models.imagenet.resnet18 \
--model torchstocks.vision.contrastive.model.resnet_simclr.Model \
--data_path /mnt/cephfs/data/stl10/ \
--image_size 96 \
--image_field feature \
--emb_size 512 \
--head_size 128 \
--eval_interval 20 \
--max_lr 4e-3 \
--num_epochs 200
```
## 实验结果
| 训练形式 |         验证方法         |         网络         | epoch | 学习率 | 数据集 | 准确率 |
| -------- | :--------------------: | :------------------: | :---: | :----: | :----: | -----: |
| 单卡     | centroid cosine distance |       resnet18       |  100  |  1e-3  | STL10  | 0.7206 |
| 单卡     | centroid cosine distance | resnet18(仅backbone) |  100  |  1e-3  | STL10  | 0.7386 |
| 单卡     | centroid cosine distance | resnet18(仅backbone) |  150  |  1e-3  | STL10  |  0.762 |
| 四卡ddp  | centroid cosine distance | resnet18(仅backbone) |  100  |  1e-3  | STL10  | 0.6951 |
| 四卡ddp  | centroid cosine distance | resnet18(仅backbone) |  100  |  4e-3  | STL10  | 0.7258 |
| 四卡ddp  | centroid cosine distance | resnet18(仅backbone) |  150  |  4e-3  | STL10  |   0.75 |
| 四卡ddp  | centroid cosine distance | resnet18(仅backbone) |  200  |  4e-3  | STL10  | 0.7706 |
| 四卡ddp  | centroid cosine distance | resnet18(仅backbone) |  300  |  4e-3  | STL10  | 0.7842 |