#!/usr/bin/env python3


import os
import random
import shlex
import shutil
import sys
from importlib import import_module

import numpy as np
import torch
from torch import nn

__all__ = [
    'fix_random_seed',
    'replace_module',
    'import_by_name',
    'load_model',
    'save_model',
    'Experiment'
]


def fix_random_seed(seed: int = 0):
    """Fix random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """Repalce module
    """
    sub_names = name.split('.')
    module = model
    for sub_name in sub_names[:-1]:
        module = getattr(module, sub_name)
    setattr(module, sub_names[-1], new_module)


def import_by_name(name: str):
    """Import by name
    """
    last_dot_idx = name.rfind('.')
    if last_dot_idx < 0:
        raise RuntimeError(f'Failed to load {name}. Only module name is given.')
    module = import_module(name[:last_dot_idx])
    member = getattr(module, name[last_dot_idx + 1:])
    return member


def load_model(model, load):
    """Load model
    """
    if load is None:
        return model
    elif isinstance(load, dict):
        for name, path in load.items():
            assert isinstance(name, str)
            assert isinstance(path, str)
            replace_module(model, name, torch.load(path))
        return model
    else:
        raise ValueError(f'Unrecognized load option: {load}.')


def save_model(model, save):
    """Save model
    """
    if save is None:
        return
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


class Experiment(object):

    def __init__(self, experiment_dir, project_dir='.'):
        self.experiment_dir = experiment_dir
        self.project_dir = project_dir

        if os.path.exists(experiment_dir):
            raise RuntimeError(f'{experiment_dir} already exists.')

        os.makedirs(experiment_dir, exist_ok=True)
        self.backup_project(project_dir, os.path.join(experiment_dir, 'src'))
        self.log('command.txt', ' '.join(map(shlex.quote, sys.argv)))

    def log(self, filename, content, end='\n'):
        with open(os.path.join(self.experiment_dir, filename), 'a') as f:
            f.write(content)
            f.write(end)

    def backup_project(self, src, dst):
        excludes = self.get_excludes(dst)
        dir_list, file_list = self.get_creation_list(root=src, excludes=excludes)
        os.makedirs(dst, exist_ok=True)
        for path in dir_list:
            os.mkdir(os.path.join(dst, os.path.relpath(path, src)))
        for path in file_list:
            target_path = os.path.join(dst, os.path.relpath(path, src))
            if os.path.getsize(path) > 1024 * 1024 * 10:
                with open(target_path, 'wb'):
                    pass
                continue
            shutil.copy(path, target_path)

    @staticmethod
    def get_creation_list(root=None, dir_list=None, file_list=None, excludes=None):
        if dir_list is None:
            dir_list = []
        if file_list is None:
            file_list = []

        for name in os.listdir(root):
            if name.startswith('.'):
                continue

            path = os.path.join(root, name) if root is not None else name
            if excludes is not None:
                if any(os.path.samefile(path, exclude) for exclude in excludes):
                    continue

            if os.path.isdir(path):
                dir_list.append(path)
                Experiment.get_creation_list(path, dir_list, file_list, excludes)
            if os.path.isfile(path):
                file_list.append(path)

        return dir_list, file_list

    @staticmethod
    def get_excludes(path):
        path = os.path.abspath(path)
        excludes = []
        while path and path != '/':
            if os.path.exists(path):
                excludes.append(path)
            path = os.path.dirname(path)
        return excludes
