#!/usr/bin/env python3

import os
import sys
from typing import Union, Iterable

import torch
import torch.distributed as dist

__all__ = [
    'RANK',
    'WORLD_SIZE',
    'DEVICE',
    'convert_device',
    'convert_model',
    'sync_grad',
    'convert_dataset'
]

from torch import nn

from torch.utils.data import Dataset

if 'RANK' in os.environ:
    dist.init_process_group('nccl')  # todo: Here, we temporary use nccl.
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    DEVICE = RANK % torch.cuda.device_count() if RANK > 0 else 'cuda'
else:
    RANK = -1
    WORLD_SIZE = 0
    DEVICE = 'cuda'

if RANK > 0:
    sys.stdout.write = lambda x: None
    sys.stderr.write = lambda x: None
    torch.save = lambda *args, **kwargs: None


def convert_device(device):
    return DEVICE if RANK >= 0 else device


def convert_model(model: nn.Module):
    if RANK >= 0:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        for p in model.parameters():
            dist.broadcast(p.data, 0)
    return model


def sync_grad(params: Union[Iterable[nn.Parameter], nn.Module]):
    if RANK >= 0:
        sync_handles = []
        sync_params = []
        if isinstance(params, nn.Module):
            params = params.parameters()
        for p in params:
            if p.grad is None:
                continue
            sync_handles.append(dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True))
            sync_params.append(p)
        for handle, p in zip(sync_handles, sync_params):
            handle.wait()
            p.grad.data /= float(WORLD_SIZE)


class DatasetWrapper(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        size = len(dataset)
        self.small_size = size // WORLD_SIZE
        self.big_size = self.small_size + 1
        num_big = size % WORLD_SIZE
        num_small = WORLD_SIZE - num_big
        self.partition = [self.big_size] * num_big + [self.small_size] * num_small
        self.start_idx = sum(self.partition[:RANK])

    def __len__(self):
        return self.big_size

    def __getitem__(self, idx):
        _idx = idx % self.partition[RANK] + self.start_idx
        return self.dataset[_idx]


def convert_dataset(dataset, is_train):
    if is_train:
        if RANK >= 0:
            return DatasetWrapper(dataset) if dataset else None
    else:
        if RANK > 0:
            return None
    return dataset
