#!/usr/bin/env python3

import os

import cv2 as cv
import onnx
import torch

from torchstocks.common.entry import AbstractModelEntry
from torchstocks.new_vision.classification.dataset import TrainDataset, TestDataset
from torchstocks.new_vision.classification.tester import ClassificationTester
from torchstocks.new_vision.classification.trainer import ClassificationTrainer
from torchstocks.utils import fix_random_seed, ArgumentParser, Config, literal_eval, import_by_name
from torchstocks.utils.image import BasicImageAugmenter

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

        self.device = self.init_device()

        self.trainer = None
        self.tester = None

    def train(self):
        args = self.args

        ################################################################################
        # create dataset
        ################################################################################
        train_dataset = None
        if args.train_path:
            aug_config = Config(BasicImageAugmenter, args)
            print(f'==== Augmenter config ====\n{aug_config}\n')
            augmenter = aug_config.build()

            train_dataset_config = Config(TrainDataset, args)
            train_dataset_config.augmenter = augmenter
            train_dataset_config.path = args.train_path
            print(f'==== Train Dataset config ====\n{train_dataset_config}\n')
            train_dataset = train_dataset_config.build()

        test_dataset = None
        if args.test_path:
            test_dataset_config = Config(TestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}\n')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(ClassificationTester, args)
        tester_config.model = self.model
        tester_config.test_dataset = test_dataset
        tester_config.device = self.device
        tester: ClassificationTester = tester_config.build()

        ################################################################################
        # create trainer
        ################################################################################
        trainer_config = Config(ClassificationTrainer, args)
        trainer_config.model = self.model
        trainer_config.train_dataset = train_dataset
        trainer_config.device = self.device

        def epoch_callback(trainer: ClassificationTrainer):
            epoch = trainer.status['epoch']
            num_epochs = trainer.status['num_epochs']
            if epoch % args.eval_every_epoch != 0 and epoch != num_epochs:
                return
            tester.run()
            trainer.status.update(tester.status)
            self.print_status(trainer.status)
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f'model.npz')
                self.save_state_dict(self.model, output_path)

        trainer_config.epoch_callback = epoch_callback
        print(f'==== Trainer config ====\n{trainer_config}\n')

        self.trainer: ClassificationTrainer = trainer_config.build()
        return self.trainer.run()

    def inference(self, doc):
        if self.tester is None:
            args = self.args
            ################################################################################
            # create dataset
            ################################################################################
            test_dataset_config = Config(TestDataset, args)
            test_dataset_config.path = None
            test_dataset = test_dataset_config.build()

            ################################################################################
            # create tester
            ################################################################################
            tester_config = Config(ClassificationTester, args)
            tester_config.model = self.model
            tester_config.test_dataset = test_dataset
            tester_config.device = self.device
            tester: ClassificationTester = tester_config.build()

            self.tester = tester

        return self.tester(doc)

    def evaluate(self):
        args = self.args
        ################################################################################
        # create dataset
        ################################################################################
        test_dataset = None
        if args.test_path:
            test_dataset_config = Config(TestDataset, args)
            test_dataset_config.path = args.test_path
            print(f'==== Test Dataset config ====\n{test_dataset_config}')
            test_dataset = test_dataset_config.build()

        ################################################################################
        # create tester
        ################################################################################
        tester_config = Config(ClassificationTester, args)
        tester_config.model = self.model
        tester_config.test_dataset = test_dataset
        tester_config.device = self.device
        tester: ClassificationTester = tester_config.build()

        self.tester = tester
        ret = tester.run()
        self.print_status(tester.status)
        return ret

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
    parser = ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--shorter_side', type=literal_eval, default=None)
    parser.add_argument('--longer_side', type=literal_eval, default=None)
    parser.add_argument('--model', default='torchstocks.new_vision.classification.model.ClassificationModel')
    parser.add_argument('--pretrained_params_file', default=None)
    parser.add_argument('--output_dir', default='saved_model_cls')

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--clip_grad_norm', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--param_groups', type=eval, default=None)
    parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval every ? epoch')

    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    entry = ModelEntry(args)
    if args.train_path:
        entry.train()
    elif args.test_path:
        entry.evaluate()
    else:
        print('Nothing to do.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
