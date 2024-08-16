#!/usr/bin/env python3


import abc
from typing import Tuple, Callable

from torch.utils.data import DataLoader, Dataset

from torchstocks import dist

__all__ = [
    'init_test_dataset',
    'AbstractTester'
]


def init_test_dataset(
        dataset: Dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=None,
        transform=None
) -> Tuple[Dataset, DataLoader, Callable]:
    if dataset is not None:
        if not (hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')):
            raise RuntimeError(f'Invalid dataset {type(dataset)}.')

        if transform is None:
            if hasattr(dataset, 'transform') and callable(dataset.transform):
                transform = dataset.transform

        if collate_fn is None:
            if hasattr(dataset, 'collate_fn') and callable(dataset.collate_fn):
                collate_fn = dataset.collate_fn

        dataset = dist.convert_dataset(dataset, is_train=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    ) if dataset is not None and len(dataset) > 0 else None

    return dataset, loader, transform


class AbstractTester(abc.ABC):

    def __init__(self):
        self._status = {}

    def set_status(self, name: str, value):
        self._status[name] = value

    def get_status(self, name: str):
        return self._status[name]

    def del_status(self, name: str):
        if name in self._status:
            del self._status[name]

    def has_status(self, name: str) -> bool:
        return name in self._status

    @property
    def status(self):
        return self._status

    init_test_dataset = staticmethod(init_test_dataset)
