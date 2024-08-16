#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-11-01
"""

import argparse
import json
import os
from json import JSONDecodeError

import cv2 as cv
import torch

from torchstocks.utils import fix_random_seed, import_by_name
from torchstocks.utils.config import Config
from torchstocks.vision.meta_learning.online_class_incremental.dataset import OnlineClassIncrementalDataset
from torchstocks.vision.meta_learning.online_class_incremental.trainer import OnlineClassIncrementalTrainer

cv.setNumThreads(0)
fix_random_seed(666)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--network', required=True, help='the network to pretrain using online meta learning')
    parser.add_argument('--load_path')
    parser.add_argument('--save_path')

    parser.add_argument('--data_path', required=True)
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1000)

    parser.add_argument(
        '--num_ways',
        type=int,
        default=1,
        help='in meta-train-train, the number of class in a inner loop'
    )
    parser.add_argument(
        '--num_shots',
        type=int,
        default=20,
        help='in meta-train-train, the number of images with every class in a inner loop',
    )
    parser.add_argument('--times_of_query', type=float, default=0)
    parser.add_argument('--random_sample_num', type=int, default=64)
    parser.add_argument(
        '--test_num_ways',
        type=int,
        default=10,
        help='in meta-test, the number of classes to eval in a sequential images',
    )
    parser.add_argument(
        '--test_num_shots',
        type=int,
        default=15,
        help='the number of image used for meta-test-train'
    )
    parser.add_argument('--test_times_of_query', type=float, default=1 / 3)
    parser.add_argument(
        '--param_groups',
        nargs='*',
        default=['{"name": "backbone"}', '{"name": "head"}']
    )
    parser.add_argument(
        '--inner_updated_layers',
        nargs='*',
        default=['head'],
    )
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--test_optimizer', default='Adam')
    parser.add_argument('--test_inner_lr', type=float, default='5e-4')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--note')
    args = parser.parse_known_args()[0]

    ################################################################################
    # create datasets
    ################################################################################
    dataset_config = Config(OnlineClassIncrementalDataset, args)
    train_dir: str = os.path.join(args.data_path, args.train_dir)
    dataset_config.path_list = [
        os.path.join(train_dir, filename)
        for filename in os.listdir(train_dir)
        if filename.endswith('.ds')
    ]
    dataset_config.train = True
    print(f'==== Train Dataset config ====\n{dataset_config}')
    train_dataset = dataset_config.build()

    test_dir: str = os.path.join(args.data_path, args.test_dir)
    dataset_config.path_list = [
        os.path.join(test_dir, filename)
        for filename in os.listdir(test_dir)
        if filename.endswith('.ds')
    ]
    dataset_config.num_ways = args.test_num_ways
    dataset_config.num_shots = args.test_num_shots
    dataset_config.times_of_query = args.test_times_of_query
    dataset_config.train = False
    print(f'==== Test Dataset Config ====\n{dataset_config}')
    test_dataset = dataset_config.build()
    ################################################################################
    # create network
    ################################################################################
    network_config = Config(import_by_name(args.network), args)
    print(f'==== Network Config ====\n{network_config}')
    network = network_config.build()
    ################################################################################
    # create model
    ################################################################################
    model_config = Config(import_by_name(args.model), args)
    model_config.network = network
    print(f'==== Model Config ====\n{model_config}')
    model = model_config.build()
    ################################################################################
    # create trainer and train
    ################################################################################
    param_groups = []
    for group in args.param_groups:
        try:
            param_groups.append(json.loads(group))
        except JSONDecodeError:
            raise ValueError(f'Invalid parameter group {group}')
    trainer_config = Config(OnlineClassIncrementalTrainer, args)
    trainer_config.model = model
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer_config.param_groups = param_groups
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==== Trainer config ====\n{trainer_config}')
    trainer = trainer_config.build()
    trainer.train()

    if args.save_path:
        print(f'Save model to {args.save_path}.')
        torch.save(model, args.save_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
