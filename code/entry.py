#!/usr/bin/env python3

import cv2 as cv
import torch

from dataset import get_dataset
from torchstocks.common.entry import AbstractModelEntry, Options, OptionItem
from torchstocks.utils import fix_random_seed, Config, import_by_name
from trainer import ClassificationTrainer

cv.setNumThreads(0)
fix_random_seed(42)

DATASET_DICT = {
    'cifar': 100,
    'caltech101': 102,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'svhn': 10,
    'sun397': 397,
    'patch_camelyon': 2,
    'eurosat': 10,
    'resisc45': 45,
    'diabetic_retinopathy': 5,
    'clevr_count': 8,
    'clevr_dist': 6,
    'dmlab': 6,
    'kitti': 4,
    'dsprites_loc': 16,
    'dsprites_ori': 16,
    'smallnorb_azi': 18,
    'smallnorb_ele': 9,
}


class TrainOptions(Options):
    dataset_name = OptionItem(required=True)
    model = OptionItem(required=True)
    image_size = OptionItem(224)
    optimizer = OptionItem('AdamW')
    batch_size = OptionItem(64)
    max_lr = OptionItem(1e-3)
    momentum = OptionItem(0.9)
    weight_decay = OptionItem(0.3)
    num_epochs = OptionItem(100)
    num_workers = OptionItem(10)
    rank = OptionItem(None, type=int)
    param_groups = OptionItem(None, type=eval)
    clip_grad_norm = OptionItem(None, type=float)


class TrainEntry(AbstractModelEntry):

    def __init__(self, args):
        super().__init__(args)

        self._init_model()
        self._init_dataset()
        self._init_trainer()

    def _init_model(self):
        args = self.args
        model_config = Config(import_by_name(args.model), args)
        model_config.num_classes = DATASET_DICT[args.dataset_name]
        print(f'==== Model config ====\n{repr(model_config)}\n')
        self.model = model_config.build()

    def _init_dataset(self):
        args = self.args
        self.train_dataset, self.test_dataset = get_dataset(args.dataset_name, args.image_size)

    def _init_trainer(self):
        args = self.args
        config = Config(ClassificationTrainer, args)
        config.model = self.model
        config.train_dataset = self.train_dataset
        config.test_dataset = self.test_dataset
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'==== Trainer config ====\n{config}\n')
        self.trainer = config.build()

    def train(self):
        self.trainer.train()
