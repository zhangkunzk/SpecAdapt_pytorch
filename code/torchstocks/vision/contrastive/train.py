#!/usr/bin/env python3

"""
python -m torchstocks.vision.contrastive.train \
    --data_path ~/data/stl10/ \
    --network torchstocks.models.imagenet.resnet18 \
    --num_classes 512 \
    --emb_size 512 \
    --head_size 128 \
    --image_size 96 \
    --image_field feature
"""

import os

import cv2 as cv
import torch

from torchstocks.utils import fix_random_seed, replace_module, import_by_name
from torchstocks.utils.config import Config, ArgumentParser, literal_eval
from torchstocks.utils.image import BasicImageAugmenter
from torchstocks.vision.contrastive.dataset import UnsupervisedDataset, SupervisedDataset
from torchstocks.vision.contrastive.trainer import ImageContrastiveTrainer

cv.setNumThreads(0)
fix_random_seed()


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--unlabeled_file', nargs='*', default=['unlabeled.ds', 'train.ds'])
    parser.add_argument('--train_file', nargs='*', default=['train.ds'])
    parser.add_argument('--test_file', nargs='*', default=['test.ds'])
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--shorter_side', type=literal_eval, default=1.1)
    parser.add_argument('--longer_side', type=literal_eval, default=None)
    parser.add_argument('--p_flip_lr', type=float, default=0.5)
    parser.add_argument('--p_color', type=float, default=0.8)
    parser.add_argument('--rnd_hue', type=float, default=0.2)
    parser.add_argument('--rnd_saturation', type=float, default=0.8)
    parser.add_argument('--rnd_brightness', type=float, default=0.8)
    parser.add_argument('--rnd_contrast', type=float, default=0.8)
    parser.add_argument('--p_grayscale', type=float, default=0.2)
    parser.add_argument('--rnd_resize', type=float, default=1.5)
    parser.add_argument('--image_field', default='image')

    parser.add_argument('--model', default='torchstocks.common.contrastive.model.simclr.SimCLRWrapper')
    parser.add_argument('--network', required=True)
    parser.add_argument('--load', type=literal_eval, default=None)
    parser.add_argument('--save', type=literal_eval, default=None)

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=30)

    parser.add_argument('--note')
    args = parser.parse_args()

    ################################################################################
    # create dataset
    ################################################################################
    unlabeled_dataset = None
    if args.unlabeled_file:
        aug_config = Config(BasicImageAugmenter, args)
        print(f'==== Augmenter config ====\n{aug_config}\n')
        augmenter = aug_config.build()

        dataset_config = Config(UnsupervisedDataset, args)
        dataset_config.augmenter = augmenter
        dataset_config.path = [os.path.join(args.data_path, path) for path in args.unlabeled_file]
        print(f'==== Unlabeled Dataset config ====\n{dataset_config}\n')
        unlabeled_dataset = dataset_config.build()

    train_dataset = None
    if args.train_file:
        dataset_config = Config(SupervisedDataset, args)
        dataset_config.path = [os.path.join(args.data_path, path) for path in args.train_file]
        train_dataset = dataset_config.build()

    test_dataset = None
    if args.test_file:
        dataset_config = Config(SupervisedDataset, args)
        dataset_config.path = [os.path.join(args.data_path, path) for path in args.test_file]
        test_dataset = dataset_config.build()

    ################################################################################
    # create model
    ################################################################################
    if isinstance(args.load, str):
        print(f'Load model from {args.load_path}.\n')
        model = torch.load(args.load_path)
    else:
        network_config = Config(import_by_name(args.network), args)
        print(f'==== Model config ====\n{repr(network_config)}\n')
        model_config = Config(import_by_name(args.model), args)
        model_config.network = network_config.build()
        print(f'==== Model config ====\n{repr(model_config)}\n')
        model = model_config.build()
        if args.load is None:
            pass
        elif isinstance(args.load, dict):
            for name, path in args.load.items():
                assert isinstance(name, str)
                assert isinstance(path, str)
                replace_module(model, name, torch.load(path))
        else:
            raise ValueError(f'Unrecognized load option: {args.load}.')

    ################################################################################
    # create trainer
    ################################################################################
    trainer_config = Config(ImageContrastiveTrainer, args)
    trainer_config.model = model
    trainer_config.unlabeled_dataset = unlabeled_dataset
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==== Trainer config ====\n{trainer_config}\n')
    trainer = trainer_config.build()
    if unlabeled_dataset:
        trainer.train()
    elif train_dataset and test_dataset:
        trainer.evaluate()
    else:
        print('Nothing to do.')

    ################################################################################
    # save model
    ################################################################################
    if args.save is None:
        pass
    elif isinstance(args.save, str):
        print(f'Save model to {args.save}.')
        torch.save(model, args.save)
    elif isinstance(args.save, dict):
        for name, path in args.save.items():
            assert isinstance(name, str)
            assert isinstance(path, str)
            torch.save(model.get_submodule(name), path)
    else:
        raise ValueError(f'Unrecognized save option: {args.save}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
