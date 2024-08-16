#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-09-17
"""

import os

import cv2 as cv
import torch

from torchstocks.utils import ArgumentParser, Config
from . import dataset
from .model import Model
from .trainer import Trainer

cv.setNumThreads(0)


class ModelEntry(object):
    """
    base_params = {
        'image_height': 320,
        'image_width': 320,
        'batch_size': 4,
        'num_workers': 10
    }

    train_params = {
        'backbone': 'wide_resnet50_2',
        'feat_size': None,
        'neighbor_size': 3,
        'mem_size': 8192,
        'num_epochs': 10
    }

    update_params = {
        "max_lr": 0.8,
        "num_epochs": 5
    }

    kwargs = {**base_params, **train_params, **update_params}
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'model_path' in self.kwargs:
            self.model = torch.load(self.kwargs['model_path'])
        else:
            model_config = Config(Model, self.kwargs)
            self.model = model_config.build()

    def train(self, task: str, train_path, test_path):
        #
        # create dataset
        image_height = self.kwargs['image_height'] if 'image_height' in self.kwargs else None
        image_width = self.kwargs['image_width'] if 'image_width' in self.kwargs else None
        hist_enhance = self.kwargs['hist_enhance'] if 'hist_enhance' in self.kwargs else False
        train_transform = dataset.TrainTransform(image_height, image_width, hist_enhance)
        if os.path.isdir(train_path):
            train_dataset = dataset.DirDataset(
                train_path,
                good_label='good',
                only_good=True,
                transform=train_transform
            )
        elif os.path.isfile(train_path):
            train_dataset = dataset.ADDataset(train_path, train_transform)
        else:
            raise RuntimeError(f'Invalid dataset {train_path}.')
        test_transform = dataset.TestTransform(image_height, image_width, hist_enhance)
        if os.path.isdir(test_path):
            test_dataset = dataset.DirDataset(
                test_path,
                good_label='good',
                transform=test_transform
            )
        elif os.path.isfile(test_path):
            test_dataset = dataset.ADDataset(test_path, test_transform)
        else:
            raise RuntimeError(f'Invalid dataset {test_path}.')

        #
        # create trainer and train
        trainer_config = Config(Trainer, self.kwargs)
        trainer_config.model = self.model
        trainer_config.train_dataset = train_dataset
        trainer_config.test_dataset = test_dataset
        trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = trainer_config.build()
        trainer.train()

    def inference(self, task: str, image):
        pass


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--image_height', type=int, default=None)
    parser.add_argument('--image_width', type=int, default=None)
    parser.add_argument('--good_label', default='good')
    parser.add_argument('--backbone', default='wide_resnet50_2')
    parser.add_argument('--aggr_kernel', type=int, default=3)
    parser.add_argument('--mem_size', type=int, default=8192)
    parser.add_argument('--feat_size', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--max_lr', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--output_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--hist_enhance', action='store_true')
    parser.add_argument('--note')
    args = parser.parse_args()

    train_transform = dataset.TrainTransform(args.image_height, args.image_width, args.hist_enhance)
    if os.path.isdir(args.train_path):
        train_dataset = dataset.DirDataset(
            args.train_path,
            good_label=args.good_label,
            only_good=True,
            transform=train_transform
        )
    elif os.path.isfile(args.train_path):
        train_dataset = dataset.ADDataset(args.train_path, train_transform)
    else:
        raise RuntimeError(f'Invalid dataset {args.train_path}.')
    test_transform = dataset.TestTransform(args.image_height, args.image_width, args.hist_enhance)
    if os.path.isdir(args.test_path):
        test_dataset = dataset.DirDataset(
            args.test_path,
            good_label=args.good_label,
            transform=test_transform
        )
    elif os.path.isfile(args.test_path):
        test_dataset = dataset.ADDataset(args.test_path, test_transform)
    else:
        raise RuntimeError(f'Invalid dataset {args.test_path}.')

    model_config = Config(Model, args)
    model = model_config.build() if args.model_path is None else torch.load(args.model_path)

    trainer_config = Config(Trainer, args)
    trainer_config.model = model
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = trainer_config.build()
    trainer.train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
