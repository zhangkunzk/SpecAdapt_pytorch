#!/usr/bin/env python3
"""
Since: 2022/11/1
Author: Howie
"""
import argparse
import os

import cv2 as cv
import torch
from docset import DocSet
from imgaug import augmenters as iaa
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from torchstocks.entries.common import import_by_name, load_extra_args
from torchstocks.utils.config import Config

from torchstocks.utils import fix_random_seed

from torchstocks.datasets.common import ImageDataset, DSDataset
from torchstocks.datasets.rehersal import RehearsalDSDataset

cv.setNumThreads(0)
fix_random_seed()


def count_num_class(train_path):
    # count classes
    max_label = -1
    with DocSet(train_path, 'r') as ds:
        for doc in tqdm(ds, leave=False, ncols=96, desc='Count Labels'):
            label = int(doc["label"])
            if label > max_label:
                max_label = label
    num_class = max_label + 1
    return num_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--load_dir_path', type=str)
    parser.add_argument('--save_dir_path', type=str)
    parser.add_argument('--sessions', type=int, required=True)
    parser.add_argument('--data_dir_path', type=str, required=True)
    parser.add_argument('--rehearsal_dir_path', type=str, default=None)
    parser.add_argument('--backbone_name', type=str, default='resnet34')
    parser.add_argument('--image_size', type=int, default=32)

    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--note')
    args = parser.parse_known_args()[0]
    os.makedirs(args.save_dir_path, exist_ok=True)
    if args.rehearsal_dir_path is None:
        args.rehearsal_dir_path = args.save_dir_path

    # create dataset
    test_datasets = []
    train_path = os.path.join(args.data_dir_path, str(args.sessions), 'train.ds')
    train_dataset = ImageDataset(
        DSDataset(train_path),
        image_field='image',
        augmenter=iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.AddToBrightness((-30, 30)),
            iaa.Pad(4, keep_size=False),
            iaa.CropToFixedSize(args.image_size, args.image_size),
            iaa.Resize((args.image_size, args.image_size), )
        ])
    )
    num_class = count_num_class(train_path)
    '''write rehearsal docset to following path'''
    rehearsal_write_path = os.path.join(
        args.rehearsal_dir_path, 'rehearse' + str(args.sessions) + '.ds'
    )

    rehearsal_path = os.path.join(
        args.rehearsal_dir_path, 'rehearse' + str(args.sessions - 1) + '.ds'
    )
    rehearsal_ds = RehearsalDSDataset(rehearsal_path, write_path=rehearsal_write_path)
    rehearsal_dataset = ImageDataset(
        rehearsal_ds,
        augmenter=iaa.Resize((args.image_size, args.image_size), )
    )

    filenames = os.listdir(args.data_dir_path)
    for filename in filenames:
        try:
            session = int(filename)
        except ValueError:
            continue
        if session <= args.sessions:
            test_path = os.path.join(args.data_dir_path, filename, 'test.ds')
            test_dataset_session = ImageDataset(
                DSDataset(test_path),
                image_field='image',
                augmenter=iaa.Sequential([
                    iaa.Resize((args.image_size, args.image_size), )
                ])
            )
            test_datasets.append(test_dataset_session)
    test_dataset = ConcatDataset(test_datasets)

    model_path = os.path.join(args.load_dir_path, f'model_{args.sessions - 1}.pth')
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model_config = Config(import_by_name(args.model))

        load_extra_args(model_config)
        model_config.load(args)
        model_config['num_class'] = num_class
        print('--------------')

        print(f'==== Model config ====\n{model_config}')
        model = model_config.build()

    trainer_config = Config(import_by_name(args.trainer))
    load_extra_args(trainer_config)
    trainer_config.load(args)
    trainer_config['model'] = model
    trainer_config['train_dataset'] = train_dataset
    trainer_config['test_dataset'] = test_dataset
    trainer_config['rehearsal_dataset'] = rehearsal_dataset
    trainer_config['num_class'] = num_class
    trainer_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==== Model config ====\n{trainer_config}')
    trainer = trainer_config.build()

    trainer.train()
    torch.save(trainer.model, os.path.join(args.save_dir_path, f'model_{args.sessions}.pth'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
