#!/usr/bin/env python3


import glob
import os
import re
from typing import Mapping

import numpy as np
import torch
from torch import nn

from torchstocks import dist

__all__ = [
    'read_state_dict',
    'read_state_dict_as_pth',
    'read_state_dict_as_npz',
    'read_state_dict_as_dir',
    'write_state_dict',
    'write_state_dict_as_pth',
    'write_state_dict_as_npz',
    'write_state_dict_as_dir',
    'load_state_dict',
    'save_state_dict',
    'init_device',
    'print_status',
    'Options',
    'OptionItem',
    'AbstractModelEntry'
]


def read_state_dict(path_or_fp) -> Mapping[str, torch.Tensor]:
    """Read a state dict form a file or folder.

    Args:
        path_or_fp: Path of the file / folder, or file-like object.

    Returns:
        A mapping represents the state dict.
    """
    if isinstance(path_or_fp, str):
        if path_or_fp.endswith('pth') or path_or_fp.endswith('pt'):
            return read_state_dict_as_pth(path_or_fp)
        elif path_or_fp.endswith('npz') or path_or_fp.endswith('zip'):
            return read_state_dict_as_npz(path_or_fp)
        else:
            return read_state_dict_as_dir(path_or_fp)
    else:
        return read_state_dict_as_npz(path_or_fp)


def read_state_dict_as_pth(path) -> Mapping[str, torch.Tensor]:
    state_dict = torch.load(path, map_location='cpu')
    if not isinstance(state_dict, Mapping):
        state_dict = state_dict.state_dict()
    assert isinstance(state_dict, Mapping)
    return state_dict


def read_state_dict_as_npz(path) -> Mapping[str, torch.Tensor]:
    array_dict = np.load(path)
    if not isinstance(array_dict, Mapping):
        raise ValueError(f'Invalid state_dict file {path}.')
    return {
        name.replace('/', '.'): torch.as_tensor(value)
        for name, value in array_dict.items()
    }


def read_state_dict_as_dir(path) -> Mapping[str, torch.Tensor]:
    state_dict = {}
    for npy_path in glob.iglob(os.path.join(path, '**/*.npy'), recursive=True):
        name = npy_path[len(path):]
        name = re.sub(r'^/+', '', name)
        if name.endswith('.npy'):
            name = name[:-4]
        name = name.replace('/', '.')
        value = np.load(npy_path)
        value = torch.as_tensor(value)
        state_dict[name] = value
    return state_dict


def write_state_dict(path_or_fp, state_dict: Mapping[str, torch.Tensor]):
    """Save a state dict to a file or folder.

    Args:
        path_or_fp: Path of the file / folder, or file-like object.
        state_dict: A mapping represents the state dict.
    """
    if isinstance(path_or_fp, str):
        if path_or_fp.endswith('.pth') or path_or_fp.endswith('pt'):
            write_state_dict_as_pth(path_or_fp, state_dict)
        elif path_or_fp.endswith('.npz'):
            write_state_dict_as_npz(path_or_fp, state_dict)
        elif path_or_fp.endswith('.zip'):
            write_state_dict_as_npz(path_or_fp[:-4], state_dict)
        else:  # save using a path, and we regard the path as a folder
            write_state_dict_as_dir(path_or_fp, state_dict)
    elif hasattr(path_or_fp, 'write'):
        write_state_dict_as_npz(path_or_fp, state_dict)
    else:
        raise ValueError(f'Invalid file type {type(path_or_fp)}.')


def write_state_dict_as_pth(path, state_dict: Mapping[str, torch.Tensor]):
    torch.save(state_dict, path)


def write_state_dict_as_npz(path, state_dict: Mapping[str, torch.Tensor]):
    state_dict = {
        name.replace('.', '/'): value.cpu().numpy()
        for name, value in state_dict.items()
    }
    np.savez(path, **state_dict)


def write_state_dict_as_dir(path, state_dict: Mapping[str, torch.Tensor]):
    for name, value in state_dict.items():
        name = name.replace('.', '/')
        value = value.cpu().numpy()
        full_path = os.path.join(path, name + '.npy')
        dir_path = os.path.dirname(full_path)
        os.makedirs(dir_path, exist_ok=True)
        np.save(full_path, value)


def load_state_dict(model: nn.Module, path_or_fp):
    loaded_state_dict = read_state_dict(path_or_fp)

    state_dict = model.state_dict()
    new_state_dict = {
        name: p for name, p in loaded_state_dict.items() if
        name in state_dict and p.shape == state_dict[name].shape
    }
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)


def save_state_dict(model: nn.Module, path_or_fp):
    state_dict = model.state_dict()
    write_state_dict(path_or_fp, state_dict)


def init_device(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = dist.convert_device(device)
    return device


def print_status(status_dict: Mapping):
    items = []

    epoch = status_dict.get('epoch')
    num_epochs = status_dict.get('num_epochs')
    if epoch is not None or num_epochs is not None:
        if epoch is None:
            epoch = '?'
        if num_epochs is None:
            num_epochs = '?'
        items.append(f'[{epoch}/{num_epochs}]')

    loss = status_dict.get('loss_g')
    if loss is None:
        loss = status_dict.get('loss')
    if loss is not None:
        items.append(f'L={loss:.06f}')

    metrics = status_dict.get('metrics')
    if metrics is not None:
        for name, value in metrics.items():
            if isinstance(value, float):
                items.append(f'{name}={value:.04f}')
            elif isinstance(value, (int, str)):
                items.append(f'{name}={value}')
            elif isinstance(value, dict):
                if 'mAP' in value:
                    mAP = value['mAP']
                    items.append(f'mAP={mAP:.02%}')
                elif 'mIoU' in value:
                    mIoU = value['mIoU']
                    items.append(f'mIoU={mIoU:.02%}')
    print(' '.join(items))


_type = type


class OptionItem(object):

    def __init__(self, default=None, type=None, required=False):
        self.default = default
        self.type = type
        self.required = required
        if self.type is None and default is not None:
            self.type = _type(default)


class Options(object):

    def option_items(self):
        for name, item in self.__class__.__dict__.items():
            if isinstance(item, OptionItem):
                yield name, item

    def check(self):
        for name, item in self.option_items():
            if name not in self.__dict__:
                setattr(self, name, item.default)
        for name, item in self.option_items():
            if getattr(self, name, None) is None and item.required:
                raise RuntimeError(f'"{name}" is required.')

    def update(self, options):
        for name, item in self.option_items():
            if hasattr(options, name):
                setattr(self, name, getattr(options, name))


class AbstractModelEntry(object):
    """Abstract model entry
    """

    def __init__(self, args):
        self.args = args
        if isinstance(self.args, Mapping):
            for name, value in self.args.items():
                if hasattr(self.args, name):
                    raise ValueError(f'Invalid option name "{name}".')
                setattr(self.args, name, value)
        elif isinstance(args, Options):
            args.check()

    load_state_dict = staticmethod(load_state_dict)
    save_state_dict = staticmethod(save_state_dict)
    init_device = staticmethod(init_device)
    print_status = staticmethod(print_status)
