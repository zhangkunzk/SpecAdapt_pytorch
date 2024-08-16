## 图像分割数据说明

图像分割数据集的标注是基于像素级别的标签，标签从0，1，2依次取值，不可间隔。若有需要忽略的像素，则按255(默认的忽略值）进行标注。一般0代表背景，每一种像素值代表一种类别，支持的标注类别最多为256类。

## 公开数据集
### CityScapes

Cityscapes是关于城市街道场景的语义理解图片数据集。它主要包含来自50个不同城市的街道场景，拥有5000张（2048 x 1024）高质量像素级注释图像，包含19个类别。Cityscapes数据集的训练集2975张，验证集500张，测试集1525张。


### Pascal VOC 2012

Pascal VOC 2012数据集以对象分割为主，包含20个类别和背景类，其中训练集1464张，验证集1449张。

通常情况下，大家会利用[SBD(Semantic Boundaries Dataset)](http://home.bharathh.info/pubs/codes/SBD/download.html)对VOC 2012数据集进行扩充，得到的训练集是10582张。


## Performance

The VOC dataset's performance is measured in terms of pixel intersection-over-union (IOU) averaged across the 21 classes.

|  Model   |  Backbone    |  Dataset   | Train resolution | Test resolution  |mIoU<sup>val<br> | max-lr| weight-decay|
|:-------- | :----------: |:----------:|:----------------:|:----------------:|:---------------:|:-----:|:-----------:|
|DeepLabV3 | ResNet50_OS8 | VOC2012    | 513 * 513        | 513 * 513        |  72.52%         |  2e-4 |  0.3        |
|DeepLabV3 | ResNet50_OS8 | VOC2012+aug| 513 * 513        | 513 * 513        |  77.69%         |  1e-3 |  0.3        |
|DeepLabV3 | ResNet50_OS8 | Cityscapes | 1024 * 1024      | 1024 * 2048      |  74.45%         |  1e-3 |  0.001      |
|DeepLabV3+| ResNet50_OS16| VOC2012    | 513 * 513        | 513 * 513        |  73.29%         |  2e-4 |  0.3        |
|DeepLabV3+| ResNet50_OS16| VOC2012+aug| 513 * 513        | 513 * 513        |  77.76%         |  1e-3 |  0.001      |


## Run the training script
CUDA_VISIBLE_DEVICES=1 python3 -m torchstocks.new_vision.segmentation.entry \
--train_path='/mnt/cephfs/data/segmentation/voc_2012_ds/train_2012.ds' \
--test_path='/mnt/cephfs/data/segmentation/voc_2012_ds/val_2012.ds' \
--image_size=513 \
--num_classes=21 \
--backbone='resnet18' \
--model=torchstocks.vision.segmentation.model.deeplabv3plus.Model


Distributed training:
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 torchrun \
--master_port=1111 \
--nproc_per_node=2 \
-m torchstocks.new_vision.segmentation.entry \
--train_path='/mnt/cephfs/data/segmentation/voc_2012_ds/train_2012.ds' \
--test_path='/mnt/cephfs/data/segmentation/voc_2012_ds/val_2012.ds' \
--backbone='resnet18' \
--image_size=513 \
--num_classes=21 \
--model=torchstocks.vision.segmentation.model.deeplabv3plus.Model

当在同一台机器上运行多个多卡任务时，通过修改master_port参数来实现.

### 模型训练、推理、导出所需参数

#### 模型超参
train_path:          str, 训练集ds文件名，可以传入多个，使用空格间隔，也可以不传，不传入任何参数时，只调用evaluate \
test_path:           测试集ds文件名，可以传入多个，使用空格间隔 \
output_dir:          str, 训练产生的模型保存地址，默认保存'best.pth'与'last.pth' \
model:               str, 指定训练使用的模型 \
backbone:            str, 主干网络名称 \
pretrained_params_file: str, 预训练模型文件 \
image_size:          int, 图像输入尺寸 \
num_classes:         int, 数据集中分割目标的种类数(包含背景类)

#### 训练超参
optimizer:           str, 优化器 \
batch_size:          int, 批大小 \
max_lr:              float, 学习率 \
momentum:            float, 动量 \
weight_decay:        float, 权重衰减 \
num_epoch:           int, 训练轮数 \
num_workers:         int, 加载数据时的进程数 \
eval_every_epoch:    int, 评估保存模型的间隔 \
param_groups:        str, 模型参数组，网络不同层设置不同参数，以deeplabv3plus模型为例，比如：'[{"name": "model.backbone", "lr": 1e-5, "mode":False}]' .将backbone的bn层封住，同时设定1e-5的学习率，其余层使用默认参数配置

#### 数据增强参数
p_scale:             float, 尺度调整因子, 0-1，默认0.5 \
p_flip_lr:           float, 水平翻转概率，0-1，默认0.5 \
rnd_rotate:          int, 旋转角度，0-90， 默认30

#### 预处理参数
image_size:          int, 图像输入的长边尺寸

#### 模型推理参数
image_size:          int, 推理时图像输入的长边尺寸

#### 模型导出参数
onnx_input_batchsize:    int, onnx模型的输入批大小, 默认为1 \
onnx_input_channel:      int, onnx模型的输入通道，默认为3 \
onnx_input_height:       int, onnx模型的输入高度，默认为513 \
onnx_input_width:        int, onnx模型的输入宽度，默认为513 \
opset:                   int, 转换时使用的onnx算子集版本,默认为12 \
save_static_onnx_file:   str, 保存的静态输入onnx文件名称 \
save_dynamic_onnx_file:  str, 保存的动态输入onnx文件名称
