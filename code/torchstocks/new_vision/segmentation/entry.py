#!/usr/bin/env python3

import os

import cv2 as cv
import onnx
import torch

from torchstocks.common.entry import AbstractModelEntry
from torchstocks.new_vision.segmentation.dataset import SegTrainDataset, SegTestDataset, SegDataCollate
from torchstocks.new_vision.segmentation.tester import SegmentationTester
from torchstocks.new_vision.segmentation.trainer import SegmentationTrainer
from torchstocks.utils.ema import ModelEMA
from torchstocks.utils import ArgumentParser, Config, fix_random_seed, import_by_name

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
        self.tester = None

    def train(self):
        """Model train
        """
        args = self.args

        ################################################################################
        # create dataset
        ################################################################################

        train_dataset = None
        if args.train_path:
            train_dataset_config = Config(SegTrainDataset, args)
            train_dataset_config.path = args.train_path
            print(f'==== Train Dataset config ====\n{train_dataset_config}\n')
            train_dataset = train_dataset_config.build()

        test_dataset = None
        if args.test_path:
            test_dataset_config = Config(SegTestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(SegmentationTester, args)
        tester_config.model = self.ema.model
        tester_config.test_dataset = test_dataset
        tester_config.test_collate = SegDataCollate()
        tester_config.device = self.device
        tester: SegmentationTester = tester_config.build()

        ################################################################################
        # create trainer
        ################################################################################
        trainer_config = Config(SegmentationTrainer, args)
        trainer_config.model = self.model
        trainer_config.ema = self.ema
        trainer_config.train_dataset = train_dataset
        trainer_config.train_collate = SegDataCollate()
        trainer_config.device = self.device

        def epoch_callback(trainer: SegmentationTrainer):
            epoch = trainer.status['epoch']
            num_epochs = trainer.status['num_epochs']
            best_mIoU = trainer.status['best_mIoU']
            if epoch % args.eval_every_epoch != 0 and epoch != num_epochs:
                return
            tester.run()
            trainer.status.update(tester.status)
            self.print_status(trainer.status)
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(self.ema.model.state_dict(), os.path.join(args.output_dir, 'last.pth'))
                if 'mIoU' in trainer.status['metrics']:
                    mIoU = trainer.status['metrics']['mIoU']
                    if mIoU > best_mIoU:
                        torch.save(self.ema.model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
                        trainer.status['best_mIoU'] = mIoU

        trainer_config.epoch_callback = epoch_callback
        print(f'==== Trainer config ====\n{trainer_config}\n')
        self.trainer: SegmentationTrainer = trainer_config.build()
        return self.trainer.run()

    def inference(self, doc):
        """Inference
        """
        if self.tester is None:
            args = self.args
            ################################################################################
            # create dataset
            ################################################################################
            test_dataset_config = Config(set, args)
            test_dataset_config.path = None
            test_dataset = test_dataset_config.build()

            ################################################################################
            # create tester
            ################################################################################
            tester_config = Config(SegmentationTester, args)
            tester_config.model = self.model
            tester_config.test_dataset = test_dataset
            tester_config.test_collate = SegDataCollate()
            tester_config.device = self.device
            tester: SegmentationTester = tester_config.build()

            self.tester = tester

        output = self.tester(doc)
        img_h, img_w = doc.shape[:2]
        res = cv.resize(output, (img_w, img_h), interpolation=cv.INTER_NEAREST)
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
            test_dataset_config = Config(SegTestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(SegmentationTester, args)
        tester_config.model = self.model
        tester_config.test_dataset = test_dataset
        tester_config.test_collate = SegDataCollate()
        tester_config.device = self.device
        tester: SegmentationTester = tester_config.build()

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
        self.load_state_dict(self.model, export_model_file)
        export_model = self.model
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
                    'outputs': {0: 'batch', 2: 'height', 3: 'width'}
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
    parser.add_argument('--image_size', type=eval, default=513)  # 支持长方形输入: h,w
    parser.add_argument('--num_classes', type=int, default=21)  # 类别数量,包含背景
    parser.add_argument('--backbone', type=str, default='resnet50')  # 主干网络
    parser.add_argument('--model', required=True)  # 模型定义
    parser.add_argument('--pretrained_params_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='saved_model_seg')

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_lr', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.93)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval every ? epoch')  # 评估保存模型的间隔

    args = parser.parse_args()

    entry = ModelEntry(args)
    if args.train_path:
        entry.train()
    elif args.test_path:
        ret = entry.evaluate()
        print(ret['metrics']['mIoU'])
    else:
        print('Nothing to do.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
