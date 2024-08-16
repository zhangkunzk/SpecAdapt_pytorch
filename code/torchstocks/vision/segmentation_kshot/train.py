#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-19
"""

import os.path

import cv2 as cv
import torch

from torchstocks.utils import ArgumentParser, Config, literal_eval, import_by_name
from torchstocks.utils.image import BasicImageAugmenter
from torchstocks.vision.segmentation_kshot.dataset import KShotSegmentationDataset
from torchstocks.vision.segmentation_kshot.trainer import SegmentationBinaryKShotTrainer

cv.setNumThreads(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--image_size', type=int, default=473)
    parser.add_argument('--shorter_side', type=literal_eval, default=None)
    parser.add_argument('--longer_side', type=literal_eval, default=1.2)
    parser.add_argument('--p_flip_lr', type=float, default=0.5)
    parser.add_argument('--rnd_hue', type=float, default=0.05)
    parser.add_argument('--rnd_saturation', type=float, default=0.2)
    parser.add_argument('--rnd_brightness', type=float, default=0.2)
    parser.add_argument('--rnd_contrast', type=float, default=0.2)
    parser.add_argument('--rnd_resize', type=float, default=0.5)
    parser.add_argument('--num_shots', type=int, default=5)

    parser.add_argument('--model', required=True)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max_lr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--param_groups', type=eval, default=[{'name': 'network.backbone', 'lr': 0}])
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--note')
    args = parser.parse_args()

    ################################################################################
    # create datasets
    ################################################################################
    aug_config = Config(BasicImageAugmenter, args)
    print(f'==== Augmenter config ====\n{aug_config}\n')
    augmenter = aug_config.build()
    dataset_config = Config(KShotSegmentationDataset, args)
    dataset_config.augmenter = augmenter
    train_dir: str = os.path.join(args.data_path, args.train_dir)
    dataset_config.train = True
    dataset_config.path_list = [
        os.path.join(train_dir, filename)
        for filename in os.listdir(train_dir)
        if filename.endswith('.ds')
    ]
    train_dataset = dataset_config.build()

    test_dir: str = os.path.join(args.data_path, args.test_dir)
    dataset_config.train = False
    dataset_config.path_list = [
        os.path.join(test_dir, filename)
        for filename in os.listdir(test_dir)
        if filename.endswith('.ds')
    ]
    test_dataset = dataset_config.build()

    ################################################################################
    # create model
    ################################################################################
    model_config = Config(import_by_name(args.model), args)
    print(f'==== Model config ====\n{repr(model_config)}\n')
    model = model_config.build()

    ################################################################################
    # create trainer and train
    ################################################################################
    trainer_config = Config(SegmentationBinaryKShotTrainer, args)
    trainer_config.model = model
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==== Trainer config ====\n{trainer_config}\n')
    trainer = trainer_config.build()

    trainer.train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
