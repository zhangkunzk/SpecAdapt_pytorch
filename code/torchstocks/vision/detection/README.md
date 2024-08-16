## 目标检测数据说明
目标检测的数据比分类复杂，一张图像中，需要标记出各个目标区域的位置和类别。
一般的目标区域位置用一个矩形框来表示，我们采用(x1,y1,x2,y2)的表达方式。

|         表达方式    |                 说明               |
| :----------------: | :--------------------------------: |
|     x1,y1,x2,y2    | (x1,y1)为左上角坐标，(x2,y2)为右下角坐标  |

### Dataset: Pascal VOC
docset格式的VOC数据放在共享盘/mnt/cephfs/data/dataset/voc目录下
|dataset         | trainset             |   |    |    | valset           |
| :----:         | :----:    | :----:   |  :----:  |  :----:  |  :----:  |
| subset         | train2007 | val2007  |train2012 | val2012  | test2007 |
|number of images| 2500      | 2509     | 5716     | 5822     | 4591     |

训练集共16547张图像，测试集4591张.

## Performance
| model               | dataset | size<br><sup>(pixels) | mAP<sup>val<br>0.5 | weight_decay|
| :-----------------: | :----:  | :----:                |  :----:            |  :--------: |
|yolov5s              | VOC     | 416                   |  79.33             |  0.3        |
|faster-rcnn(resnet18)| VOC     | 640                   |  79.21%            |  0.001      |
|faster-rcnn(resnet50)| VOC     | 640                   |  82.62%            |  0.001      |

## Run the training script

### Yolo model 单卡
CUDA_VISIBLE_DEVICES=1 python3 -m torchstocks.vision.detection.train \
--train_data_path='/mnt/cephfs/data/dataset/voc/train_all_data' \
--train_file='' \
--test_data_path='/mnt/cephfs/data/dataset/voc/' \
--test_file='test2007.ds' \
--image_size=416 \
--num_classes=20 \
--model=torchstocks.vision.detection.model.yolo.Model \
--feat_size 32 \
--num_bottlenecks 1 \
--p_scale 0.5 \
--p_translate 0.1 \
--p_flip_lr 0.5 \
--mosaic True \
--train_obj_threshold 0.001 \
--train_nms_threshold 0.6 \
--test_obj_threshold 0.25 \
--test_nms_threshold 0.45 \
--vis_threshold 0.2 \
--onnx_input_batchsize 1 \
--onnx_input_channle 3 \
--onnx_input_height 640 \
--onnx_input_width 640 \
--opset 12 \
--save_static_onnx_file 'yolov5s_static.onnx' \
--save_dynamic_onnx_file 'yolov5s_dynamic.onnx'

### Yolo model 多卡
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 torchrun \
--master_port=1111 \
--nproc_per_node=2 \
-m torchstocks.vision.detection.train \
--model=torchstocks.vision.detection.model.yolo.Model \
--image_size=416 \
--num_classes=20 \
--train_data_path='/mnt/cephfs/data/dataset/voc/train_all_data' \
--train_file='' \
--test_data_path='/mnt/cephfs/data/dataset/voc/' \
--test_file='test2007.ds'
当在同一台机器上运行多个多卡任务时，通过修改master_port参数来实现.

### Faster-RCNN model
CUDA_VISIBLE_DEVICES=1 python3 -m torchstocks.vision.detection.train \
--model=torchstocks.vision.detection.model.faster_rcnn.Model \
--image_size=640 \
--num_classes=20 \
--train_data_path='/mnt/cephfs/data/dataset/voc/train_all_data' \
--train_file='' \
--test_data_path='/mnt/cephfs/data/dataset/voc/' \
--test_file='test2007.ds' \
--weight_decay=0.001 \
--obj_threshold=0.05 \
--num_epochs=100


### 模型训练、推理、导出所需参数

#### 模型超参
train_data_path:     str, 训练集ds文件路径  \
train_file:          str, 训练集ds文件名, 当训练集含有多个ds文件时, 将其放入同一个文件夹内, train_file设置为空 \
test_data_path:      str, 测试集ds文件路径 \
test_file:           str, 测试集ds文件名, 当测试集含有多个ds文件时, 将其放入同一个文件夹内, test_file设置为空 \
sort_testset:        可选参数，是否对测试集按照图像大小排序，一般不需要
output_dir:          str, 训练产生的模型保存地址，默认保存'best.pth'与'last.pth' \
model:               str, 指定训练使用的模型 \
backbone:            str, 主干网络名称，检测任务中faster-rcnn网络需要设置，yolo网络不需要 \
pretrained_params_file: str, 预训练模型文件 \
image_size:          int, 图像输入尺寸 \
num_classes:         int, 数据集中检测目标的种类数(yolo模型不包含背景类) \
feat_size:           int, 网络宽度，用于调整yolo模型大小 \
num_bottlenecks:     int, 网络深度，用于调整yolo模型大小 \
train_obj_threshold: float, 训练过程中的目标阈值，默认0.001, 用于判断检测框是否是前景目标 \
train_nms_threshold: float, 训练过程中的nms阈值，默认0.6, 用于过滤重合过多的检测框

#### 训练超参
optimizer:           str, 优化器 \
batch_size:          int, 批大小 \
max_lr:              float, 学习率 \
momentum:            float, 动量 \
weight_decay:        float, 权重衰减 \
num_epoch:           int, 训练轮数 \
num_workers:         int, 加载数据时的进程数 \
eval_every_epoch:    int, 评估保存模型的间隔 \
param_groups:        str, 模型参数组，网络不同层设置不同参数，以yolo模型为例，比如：'[{"names": ["model.backbone", "model.fpn", "model.pan"], "lr": 1e-5, "mode":False}]' .将backbone, fpn, pan的bn层封住，同时设定1e-5的学习率，其余层使用默认参数配置

#### 数据增强参数
p_scale:             float, 尺度调整因子, 0-1，默认0.5 \
p_translate:         float, 平移比例，0-1，默认0.1 \
p_flip_lr:           float, 水平翻转概率，0-1，默认0.5 \
mosaic:              bool, 是否使用mosaic数据增强，默认为True

#### 预处理参数
image_size:          int, 图像输入的长边尺寸

#### 模型推理参数
image_size:          int, 推理时图像输入的长边尺寸 \
test_obj_threshold:  float, 推理过程中的目标阈值，默认0.25，用于判断检测框是否是前景目标 \
test_nms_threshold:  float, 推理过程中的nms阈值，默认0.45，用于过滤重合过多的检测框 \
vis_threshold:       float, 可视化检测结果时的阈值，默认0.2

#### 模型导出参数
onnx_input_batchsize:    int, onnx模型的输入批大小, 默认为1 \
onnx_input_channel:      int, onnx模型的输入通道，默认为3 \
onnx_input_height:       int, onnx模型的输入高度，默认为640 \
onnx_input_width:        int, onnx模型的输入宽度，默认为640 \
opset:                   int, 转换时使用的onnx算子集版本,默认为12 \
save_static_onnx_file:   str, 保存的静态输入onnx文件名称 \
save_dynamic_onnx_file:  str, 保存的动态输入onnx文件名称

