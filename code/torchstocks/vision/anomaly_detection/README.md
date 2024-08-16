# Anomaly Detection

## Usage

```bash
python -m torchstocks.vision.anomaly_detection.train \
--train_path train.ds \
--test_path test.ds \
--image_width 320 \
--image_height 320 \
--backbone wide_resnet101_2 \
--feat_size 512 \
--mem_size 4096 \
--use_amp
```