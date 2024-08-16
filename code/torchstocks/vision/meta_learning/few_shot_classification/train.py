#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-11-12
"""

import os

import cv2 as cv
import torch

from torchstocks.utils import fix_random_seed, ArgumentParser, Config, literal_eval, replace_module, import_by_name
from torchstocks.vision.meta_learning.few_shot_classification.dataset import FewShotDateset
from torchstocks.vision.meta_learning.few_shot_classification.trainer import FewShotTrainer

cv.setNumThreads(0)
fix_random_seed(6)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--load', default=None)
    parser.add_argument('--save', default=None)

    parser.add_argument('--data_path', required=True)
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--image_size', type=int, default=84)

    parser.add_argument('--num_ways', type=int, default=5)
    parser.add_argument('--num_shots', type=int, default=5)

    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--param_groups', type=eval, default=None)
    parser.add_argument('--clip_grad_norm', type=float, default=0.1)
    parser.add_argument('--note')
    args = parser.parse_args()
    ################################################################################
    # create datasets
    ################################################################################
    dataset_config = Config(FewShotDateset, args)
    train_dir = os.path.join(args.data_path, args.train_dir)
    dataset_config.path_list = [
        os.path.join(train_dir, filename)
        for filename in os.listdir(train_dir)
        if filename.endswith('.ds')
    ]
    dataset_config.train = True
    print(f'==== Train Dataset config ====\n{dataset_config}')
    train_dataset = dataset_config.build()
    test_dir = os.path.join(args.data_path, args.test_dir)
    dataset_config.path_list = [
        os.path.join(test_dir, filename)
        for filename in os.listdir(test_dir)
        if filename.endswith('.ds')
    ]
    dataset_config.train = False
    print(f'==== Test Dataset Config ====\n{dataset_config}')
    test_dataset = dataset_config.build()
    ################################################################################
    # create model
    ################################################################################
    load = literal_eval(args.load)
    if isinstance(load, str):
        print(f'Load model from {args.load_path}.')
        model = torch.load(args.load_path)
    else:
        network_config = Config(import_by_name(args.network), args)
        network_config.num_class = args.num_ways
        print(f'==== Network config ====\n{repr(network_config)}\n')
        network = network_config.build()
        model_config = Config(import_by_name(args.model), args)
        model_config.network = network
        print(f'==== Model config ====\n{repr(model_config)}\n')
        model = model_config.build()
        if load is None:
            pass
        elif isinstance(load, dict):
            for name, path in load.items():
                assert isinstance(name, str)
                assert isinstance(path, str)
                replace_module(model, name, torch.load(path))
        else:
            raise ValueError(f'Unrecognized load option: {load}.')
    ################################################################################
    # create trainer and train
    ################################################################################
    trainer_config = Config(FewShotTrainer, args)
    trainer_config.model = model
    trainer_config.train_dataset = train_dataset
    trainer_config.test_dataset = test_dataset
    trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==== Trainer config ====\n{trainer_config}')
    trainer = trainer_config.build()
    trainer.train()
    ################################################################################
    # save model
    ################################################################################
    save = literal_eval(args.save)
    if save is None:
        pass
    elif isinstance(save, str):
        print(f'Save model to {save}.')
        torch.save(model, save)
    elif isinstance(save, dict):
        for name, path in save.items():
            assert isinstance(name, str)
            assert isinstance(path, str)
            torch.save(model.get_submodule(name), path)
    else:
        raise ValueError(f'Unrecognized save option: {save}.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
