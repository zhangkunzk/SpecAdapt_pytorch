#!/usr/bin/env python3

import argparse
import os
import collections
import cv2 as cv
import numpy as np
import random
import torch
from docset import DocSet
from torch.utils.data import ConcatDataset
from tqdm import tqdm

cv.setNumThreads(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


set_seed(37)

from torchstocks.vision.class_incremental.dataset import TrainDataset, TestDataset
from torchstocks.common.dataset import RehearsalDSDataset
from torchstocks.vision.class_incremental import incremental


def count_classes(train_path):
    train_class_dict = collections.defaultdict(list)
    all_doc = DocSet(train_path, 'r')
    for doc_i in all_doc:
        train_class_dict[int(doc_i['label'])].append(doc_i)
    class_name = list(train_class_dict.keys())
    return max(class_name) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--rehearsal-path', default=None)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--sessions', type=int, required=True)
    parser.add_argument('--alg', default='lwf')
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.93)
    parser.add_argument('--weight-decay', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=100)

    parser.add_argument('--workspace', default='output')
    parser.add_argument('--note')
    args = parser.parse_args()

    os.makedirs(args.workspace, exist_ok=True)
    for cur_session in range(0, 10):
        print('----------------')
        print('algorithm: ', args.alg)
        print('session: ', cur_session)
        print('backbone: ', args.backbone)
        print('batch size', args.batch_size)
        print('max lr: ', args.max_lr, 'weight decay: ', args.weight_decay)
        print('note: ', args.note)
        print('---------------')
        # create dataset
        test_datasets = []
        rehearsal_dataset = None
        train_path = os.path.join(args.data_path, str(cur_session), 'train.ds')
        train_dataset = TrainDataset(path=train_path, image_size=args.image_size)

        '''write rehearsal docset to following path'''
        rehearsal_write_path = os.path.join(
            args.rehearsal_path, 'rehearse' + str(cur_session) + '.ds'
        )

        if args.rehearsal_path:
            rehearsal_path = os.path.join(
                args.rehearsal_path, 'rehearse' + str(cur_session - 1) + '.ds'
            )
            # if os.path.exists(rehearsal_path):
            rehearsal_ds = RehearsalDSDataset(rehearsal_path, write_path=rehearsal_write_path)
            rehearsal_dataset = TrainDataset(path=None, image_size=args.image_size, rehearsal_dataset=rehearsal_ds)

        filenames = os.listdir(args.data_path)
        for filename in filenames:
            try:
                session = int(filename)
            except ValueError:
                continue
            if session <= cur_session:
                test_path = os.path.join(args.data_path, filename, 'test.ds')
                test_datasets.append(TestDataset(path=test_path, image_size=args.image_size))
        test_dataset = ConcatDataset(test_datasets)

        # count classes
        max_label = -1
        with DocSet(train_path, 'r') as ds:
            for doc in tqdm(ds, leave=False, ncols=96, desc='Count Labels'):
                label = int(doc["label"])
                if label > max_label:
                    max_label = label
        num_class = max_label + 1

        # create model
        alg = getattr(incremental, args.alg)
        Trainer = alg.AlgTrainer
        Model = alg.AlgModel
        if cur_session == 0:
            model = Model(args.backbone, num_class)
        else:
            model = torch.load(os.path.join(args.workspace, f'model_{cur_session - 1}.pth'))

        ce_weight = 0.0001 - 0.00001 * (cur_session - 1)
        # create trainer
        trainer = Trainer(
            model,
            train_dataset,
            test_dataset,
            rehearsal_dataset,
            num_class,
            args.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        for name in dir(trainer):
            if not name.startswith('_') and hasattr(args, name):
                setattr(trainer, name, getattr(args, name))
        trainer.run_train()
        torch.save(trainer.model, os.path.join(args.workspace, f'model_{cur_session}.pth'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
