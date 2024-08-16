# Run script

```bash
CUDA_VISIBLE_DEVICES='3' python -m torchstocks.vision.meta_learning.few_shot_classification.train --data_path /mnt/cephfs/data/miniimagenet_class/ --model torchstocks.vision.meta_learning.few_shot_classification.model.maml.Model --network torchstocks.vision.meta_learning.few_shot_classification.model.maml.ConvNet
```