#!/usr/bin/env python3
"""
Since: 2022/10/31
Author: Howie
"""

import collections
import glob
import os
import random
from typing import Iterable, Mapping, Callable, MutableMapping, Sized
from typing import Union, Sequence

import numpy as np
import torch
from docset import DocSet, ConcatDocSet
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.datasets.common import DSDataset, ImageDataset
from torchstocks.utils.image import RandomCropToSquare

__all__ = [
    'RehearsalDSDataset'
]


class RehearsalDSDataset(Dataset):

    def __init__(self,
                 path: Union[str, Sequence[str]],
                 write_path: str,
                 label_field: str = 'label',
                 samples_pre_class: int = 20
                 ):

        self.label_field = label_field
        self.samples_pre_class = samples_pre_class
        self.write_path = write_path
        if os.path.exists(path):
            self.ds = DocSet(path)
        else:
            self.ds = None

    def __len__(self):
        if self.ds:
            return len(self.ds)
        else:
            return 0

    def __getitem__(self, i):
        return self.ds[i]

    def write(self, dataset, num_class):
        sampling_idx = [[] for _ in range(num_class)]
        for idx, doc in enumerate(dataset):
            class_ = doc[self.label_field]
            if len(sampling_idx[class_]) < self.samples_pre_class:
                sampling_idx[class_].append(idx)
            else:
                random_position = random.randint(0, self.samples_pre_class - 1)
                sampling_idx[class_][random_position] = idx
        with DocSet(self.write_path, 'w') as ds:
            for i, idx_list_i in enumerate(sampling_idx):
                for j in idx_list_i:
                    ds.write(dataset[j])
            if self.ds is not None:
                for doc_i in self.ds:
                    ds.write(doc_i)



