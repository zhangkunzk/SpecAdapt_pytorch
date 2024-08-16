#!/usr/bin/env python3

import os

from copy import deepcopy
import cv2 as cv
import onnx
import torch

from torchstocks.common.entry import AbstractModelEntry
from torchstocks.new_vision.detection import dataset
from torchstocks.new_vision.detection.dataset import DetTrainDataset, DetTestDataset, DetDataCollate
from torchstocks.new_vision.detection.tester import DetectionTester
from torchstocks.new_vision.detection.trainer import DetectionTrainer
from torchstocks.utils.ema import ModelEMA
from torchstocks.utils import ArgumentParser, Config, fix_random_seed, import_by_name
from torchstocks.vision.detection.model.yolov5 import YoloDecoder
from torchstocks.vision.detection.model.rcnn.decoder import FastRCNNDecoder


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
        self.model = model_config.build()
        if args.pretrained_params_file:
            self.load_state_dict(self.model, args.pretrained_params_file)
        self.ema = ModelEMA(self.model)

        self.device = self.init_device()

        self.trainer = None
        self.decoder = None
        self.tester = None

        ################################################################################
        # create decoder and data format
        ################################################################################
        # maybe need to optimize
        if hasattr(self.model, 'num_heads'):  # yolo
            yolo_format = True
            rcnn_format = False
            data_format = 'xywh'
            pct = True
            decoder = YoloDecoder(
                obj_threshold=args.train_obj_threshold,
                nms_threshold=args.train_nms_threshold
            )
        elif hasattr(self.model, 'in_features'):  # faster_rcnn
            yolo_format = False
            rcnn_format = True
            data_format = 'xyxy'
            pct = False
            decoder = FastRCNNDecoder(
                score_threshold=args.train_obj_threshold,
                nms_threshold=args.train_nms_threshold,
                topk_per_image=100
            )
        else:
            print('Please check your model')

        self.decoder = decoder.to(self.device)

        dataset.YOLO_FORMAT = yolo_format
        dataset.RCNN_FORMAT = rcnn_format
        dataset.DATA_FORMAT = data_format
        dataset.PCT = pct

    def train(self):
        """Model train
        """
        args = self.args

        ################################################################################
        # create dataset
        ################################################################################

        train_dataset = None
        if args.train_path:
            train_dataset_config = Config(DetTrainDataset, args)
            train_dataset_config.path = args.train_path
            print(f'==== Train Dataset config ====\n{train_dataset_config}\n')
            train_dataset = train_dataset_config.build()

        test_dataset = None
        if args.test_path:
            test_dataset_config = Config(DetTestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(DetectionTester, args)
        tester_config.model = self.ema.model
        tester_config.decoder = self.decoder
        tester_config.test_dataset = test_dataset
        tester_config.test_collate = DetDataCollate()
        tester_config.device = self.device
        tester: DetectionTester = tester_config.build()

        ################################################################################
        # create trainer
        ################################################################################
        trainer_config = Config(DetectionTrainer, args)
        trainer_config.model = self.model
        trainer_config.ema = self.ema
        trainer_config.train_dataset = train_dataset
        trainer_config.train_collate = DetDataCollate()
        trainer_config.device = self.device

        def epoch_callback(trainer: DetectionTrainer):
            epoch = trainer.status['epoch']
            num_epochs = trainer.status['num_epochs']
            best_mAP = trainer.status['best_mAP']
            if epoch % args.eval_every_epoch != 0 and epoch != num_epochs:
                return
            tester.run()
            trainer.status.update(tester.status)
            self.print_status(trainer.status)
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(self.ema.model, os.path.join(args.output_dir, 'last.pth'))
                if 'mAP' in trainer.status['metrics']['AP50']:
                    mAP = trainer.status['metrics']['AP50']['mAP']
                    if mAP > best_mAP:
                        torch.save(self.ema.model, os.path.join(args.output_dir, 'best.pth'))
                        trainer.status['best_mAP'] = mAP

        trainer_config.epoch_callback = epoch_callback
        print(f'==== Trainer config ====\n{trainer_config}\n')
        self.trainer: DetectionTrainer = trainer_config.build()
        return self.trainer.run()

    @staticmethod
    def post_process(output, img_h, img_w, obj_thres=0.25):
        '''
        output: a (n, 6) tensor per image [cx, cy, w, h, cls, conf]
        '''
        output = output[torch.where(output[:, -1] > obj_thres)]
        res = deepcopy(output)
        res[:, 0] = img_w * (output[:, 0] - output[:, 2] / 2)  # x1 = cx-w/2
        res[:, 1] = img_h * (output[:, 1] - output[:, 3] / 2)  # y1 = cy-h/2
        res[:, 2] = img_w * (output[:, 0] + output[:, 2] / 2)  # x2 = cx+w/2
        res[:, 3] = img_h * (output[:, 1] + output[:, 3] / 2)  # y2 = cy+h/2
        return res.cpu().numpy()

    def inference(self, doc):
        """Inference
        """

        args = self.args
        ################################################################################
        # create decoder
        ################################################################################
        # maybe need to optimize
        if hasattr(self.model, 'num_heads'):  # yolo
            decoder = YoloDecoder(
                obj_threshold=args.test_obj_threshold,
                nms_threshold=args.test_nms_threshold
            )
        elif hasattr(self.model, 'in_features'):  # faster_rcnn
            decoder = FastRCNNDecoder(
                score_threshold=args.test_obj_threshold,
                nms_threshold=args.test_nms_threshold,
                topk_per_image=100
            )
        else:
            print('Please check your model')
        decoder = decoder.to(self.device)

        if self.tester is None:
            ################################################################################
            # create dataset
            ################################################################################
            test_dataset_config = Config(DetTestDataset, args)
            test_dataset_config.path = None
            test_dataset_config.pad_flag = False
            test_dataset = test_dataset_config.build()

            ################################################################################
            # create tester
            ################################################################################
            tester_config = Config(DetectionTester, args)
            tester_config.model = self.model
            tester_config.decoder = decoder
            tester_config.test_dataset = test_dataset
            tester_config.test_collate = DetDataCollate()
            tester_config.device = self.device
            tester: DetectionTester = tester_config.build()
            self.tester = tester
        else:
            self.tester.decoder = decoder

        output = self.tester(doc)
        img_h, img_w = doc.shape[:2]
        res = self.post_process(output, img_h, img_w)
        return res

    def evaluate(self):
        """Evaluate
        """
        args = self.args
        ################################################################################
        # create dataset
        ################################################################################
        test_dataset = None
        if args.test_path:
            test_dataset_config = Config(DetTestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(DetectionTester, args)
        tester_config.model = self.model
        tester_config.decoder = self.decoder
        tester_config.test_dataset = test_dataset
        tester_config.test_collate = DetDataCollate()
        tester_config.device = self.device
        tester: DetectionTester = tester_config.build()

        self.tester = tester
        tester.run()
        self.print_status(tester.status)
        return tester.status

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
        export_model = torch.load(
            export_model_file, map_location='cpu').to(self.device)
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
                    'outputs': {0: 'batch'}
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
    parser.add_argument('--train_path', type=str, nargs='+')
    parser.add_argument('--test_path', type=str, nargs='+')
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pretrained_params_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='saved_model_det')
    parser.add_argument('--train_obj_threshold', type=float, default=0.001)
    parser.add_argument('--train_nms_threshold', type=float, default=0.6)
    parser.add_argument('--test_obj_threshold', type=float, default=0.25)
    parser.add_argument('--test_nms_threshold', type=float, default=0.4)

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--clip_grad_norm', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--param_groups', type=eval, default=None)
    parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval every ? epoch')

    args = parser.parse_args()

    entry = ModelEntry(args)
    if args.train_path:
        entry.train()
    elif args.test_path:
        ret = entry.evaluate()
        print(ret['metrics']['AP50']['mAP'])
    else:
        print('Nothing to do.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
