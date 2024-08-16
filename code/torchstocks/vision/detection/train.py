#!/usr/bin/env python3

import os
from typing import Sequence
from copy import deepcopy

import cv2 as cv
import numpy as np
import torch
import onnx

from torchstocks.common.entry import AbstractModelEntry
from torchstocks.utils import import_by_name, fix_random_seed, ArgumentParser
from torchstocks.utils.config import Config
from torchstocks.vision.detection.dataset import DetTrainDataset, DetTestSortedDataset
from torchstocks.vision.detection.dataset import DetTestDataset, TestTransform, InferenceTransform
from torchstocks.vision.detection.dataset import DetDataCollate, DetTestSortedDataCollate
from torchstocks.vision.detection.trainer import DetectionTrainer
from torchstocks.vision.detection import dataset
from torchstocks.vision.detection.model.yolov5 import YoloDecoder


cv.setNumThreads(0)
fix_random_seed()


class ModelEntry(AbstractModelEntry):
    """Model entry
    """

    def __init__(self, args):
        super().__init__(args)

        ################################################################################
        # create model
        ################################################################################
        model_config = Config(import_by_name(args.model), args)
        print(f'==== Model config ====\n{repr(model_config)}\n')
        model = model_config.build()
        self.model = model
        if args.pretrained_params_file:
            assert os.path.exists(args.pretrained_params_file)
            pretrained_params = torch.load(args.pretrained_params_file, map_location='cpu')
            if not isinstance(pretrained_params, dict):
                pretrained_params = pretrained_params.state_dict()
            assert isinstance(pretrained_params, dict)
            model_state_dict = model.state_dict()
            params = {
                k: v for k, v in pretrained_params.items() if
                k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            model_state_dict.update(params)
            self.model.load_state_dict(model_state_dict)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.test_transform = Config(TestTransform, args).build()
        self.inference_transform = None

    def train(self):
        """Model train
        """
        args = self.args

        ################################################################################
        # create dataset
        ################################################################################
        # maybe need to optimize
        if hasattr(self.model, 'num_heads'):
            yolo_format = True
            rcnn_format = False
            data_format = 'xywh'
            pct = True
        elif hasattr(self.model, 'in_features'):
            yolo_format = False
            rcnn_format = True
            data_format = 'xyxy'
            pct = False
        else:
            print('Please check your model')

        dataset.YOLO_FORMAT = yolo_format
        dataset.RCNN_FORMAT = rcnn_format
        dataset.DATA_FORMAT = data_format
        dataset.PCT = pct

        train_dataset = None
        if args.train_file or args.train_data_path:
            train_dataset_config = Config(DetTrainDataset, args)
            train_dataset_config.path = os.path.join(args.train_data_path, args.train_file)
            print(f'==== Train Dataset config ====\n{train_dataset_config}')
            train_dataset = train_dataset_config.build()

        test_dataset = None
        if args.test_file or args.test_data_path:
            if args.sort_testset:
                print('********sort test dataset********')
                test_dataset_config = Config(DetTestSortedDataset, args)
            else:
                test_dataset_config = Config(DetTestDataset, args)  # collate_fn与DetDataCollate对应
            test_dataset_config.path = os.path.join(args.test_data_path, args.test_file)
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create trainer
        ################################################################################
        trainer_config = Config(DetectionTrainer, args)
        trainer_config.model = self.model
        trainer_config.train_dataset = train_dataset
        trainer_config.test_dataset = test_dataset
        trainer_config.train_collate = DetDataCollate()
        if args.sort_testset:
            trainer_config.test_collate = DetTestSortedDataCollate(
                image_size=args.image_size
            )
        else:
            trainer_config.test_collate = DetDataCollate()
        trainer_config.device = self.device
        print(f'==== Train trainer config ====\n{trainer_config}')
        trainer = trainer_config.build()
        return trainer, train_dataset, test_dataset

    @staticmethod
    def post_process(output, img_h, img_w, vis_thres=0.2):
        '''
        output: a (n, 6) tensor per image [cx, cy, w, h, cls, conf]
        '''
        output = output[torch.where(output[:, -1] > vis_thres)]
        res = deepcopy(output)
        res[:, 0] = img_w * (output[:, 0] - output[:, 2] / 2)  # x1 = cx-w/2
        res[:, 1] = img_h * (output[:, 1] - output[:, 3] / 2)  # y1 = cy-h/2
        res[:, 2] = img_w * (output[:, 0] + output[:, 2] / 2)  # x2 = cx+w/2
        res[:, 3] = img_h * (output[:, 1] + output[:, 3] / 2)  # y2 = cy+h/2
        return res.cpu().numpy()

    def do_inference(self, infer_model, decoder, image):
        """Inference one image
        """
        img_h, img_w = image.shape[:2]
        resized_image = self.inference_transform(image)  # keep aspect ratio resize
        input_tensor = torch.from_numpy(resized_image).unsqueeze(0).to(self.device)
        output = decoder(infer_model(input_tensor))[0]
        result = self.post_process(output, img_h, img_w, vis_thres=self.args.vis_threshold)
        return result

    def inference(self, infer_model_file, input_image):
        """Runs inference results

        Args:
            infer_model_file: pytorch model file for inference
            input_image: ndarray or ndarray sequence, rgb format

        Returns:
            results: ndarray or ndarray sequence, for per image [x1, y1, x2, y2, cls, conf]
        """
        self.inference_transform = InferenceTransform(image_size=self.args.image_size)
        try:
            assert os.path.exists(infer_model_file)
            infer_model = torch.load(infer_model_file, map_location='cpu').to(self.device)
            if hasattr(infer_model, 'num_heads'):
                decoder = YoloDecoder(
                    obj_threshold=self.args.test_obj_threshold,
                    nms_threshold=self.args.test_nms_threshold
                )
        except Exception as e:
            print(e)

        if isinstance(input_image, Sequence):
            results = []
            for image in input_image:
                results.append(self.do_inference(infer_model, decoder, image))
        elif isinstance(input_image, np.ndarray):
            results = self.do_inference(infer_model, decoder, input_image)
        else:
            print('Please check your input data!')
        return results

    def export_onnx(self, export_model_file):
        """Export pytorch model to onnx

        Args:
            export_model_file: pytorch model file for export

        Returns:

        """
        assert self.args.onnx_input_batchsize > 0, self.args.onnx_input_channel == 3
        assert self.args.onnx_input_height > 0, self.args.onnx_input_width > 0
        assert self.args.save_static_onnx_file, self.args.save_dynamic_onnx_file
        assert self.args.opset >= 12
        export_model = torch.load(export_model_file, map_location='cpu').to(self.device)
        export_model.eval()
        static_dummy_input = torch.randn(
            self.args.onnx_input_batchsize,
            self.args.onnx_input_channel,
            self.args.onnx_input_height,
            self.args.onnx_input_width
        ).to(self.device)
        try:
            print(f'Starting export with onnx {onnx.__version__}...')
            print('Starting export static onnx file...')
            torch.onnx.export(
                export_model,
                static_dummy_input,
                self.args.save_static_onnx_file,
                verbose=False,
                opset_version=self.args.opset,
                input_names=['inputs'],
                output_names=['outputs'],
                dynamic_axes=None
            )
            print('Checking generated static onnx file...')
            model_onnx = onnx.load(self.args.save_static_onnx_file)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            print(f'Export success, saved as {self.args.save_static_onnx_file}')

            print('Starting export dynamic onnx file...')
            torch.onnx.export(
                export_model,
                static_dummy_input,
                self.args.save_dynamic_onnx_file,
                verbose=False,
                opset_version=self.args.opset,
                input_names=['inputs'],
                output_names=['outputs'],
                dynamic_axes={
                    'inputs': {0: 'batch', 2: 'height', 3: 'width'},
                    'outputs': {0: 'batch', 1: 'anchors'}
                }
            )
            print('Checking generated dynamic onnx file...')
            model_onnx = onnx.load(self.args.save_dynamic_onnx_file)
            onnx.checker.check_model(model_onnx)
            print(f'Export success, saved as {self.args.save_dynamic_onnx_file}')

        except Exception as e:
            print(f'Export failure: {e}')


def main():
    """Main
    """
    parser = ArgumentParser()
    parser.add_argument('--train_data_path', required=True)  # 训练集数据路径
    parser.add_argument('--test_data_path', required=True)   # 测试集数据路径
    parser.add_argument('--train_file', default='train.ds')  # 训练集文件名称
    parser.add_argument('--test_file', default='test.ds')    # 测试集文件名称
    parser.add_argument('--sort_testset', action='store_true')  # 是否对测试集数据按照图像大小排序
    parser.add_argument('--image_size', type=int, default=416)  # 训练时传入网络的图像大小
    parser.add_argument('--num_classes', type=int, default=20)  # 类别数量
    parser.add_argument('--backbone', type=str, default='resnet18')  # 主干网络名称
    parser.add_argument('--model', required=True)  # 模型定义
    parser.add_argument('--pretrained_params_file', default=None)  # 预训练模型文件
    parser.add_argument('--output_dir', default='saved_model')  # 训练过程中模型保存的路径
    parser.add_argument('--param_groups', type=eval, default=None)  # 网络参数分组

    parser.add_argument('--optimizer', default='AdamW')         # 优化器
    parser.add_argument('--batch_size', type=int, default=32)   # 批大小
    parser.add_argument('--max_lr', type=float, default=1e-3)   # 学习率
    parser.add_argument('--momentum', type=float, default=0.93) # 动量
    parser.add_argument('--weight_decay', type=float, default=0.3) # 权值衰减
    parser.add_argument('--num_epochs', type=int, default=300)  # 训练轮数
    parser.add_argument('--num_workers', type=int, default=10)  # 加载数据时的进程数
    parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval every ? epoch')  # 评估保存模型的间隔

    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    trainer, train_dataset, test_dataset = ModelEntry(args).train()
    if train_dataset:
        trainer.train()
        progress = trainer.get_status('progress')  # 0~1之间
        print('progress: ', progress)
        if 'metrics' in trainer.status:  # 训练eval_every_epoch个epoch之后才会计算一次metrics
            scores = trainer.get_status('metrics')
            print(scores)
    elif test_dataset:
        trainer.evaluate()
    else:
        print('Nothing to do.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
