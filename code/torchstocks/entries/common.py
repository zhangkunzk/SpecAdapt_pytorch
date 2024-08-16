#!/usr/bin/env python3

"""
@author: xi
@since: 2022-10-20
"""

import argparse
import collections
from importlib import import_module
from typing import Literal

from torchstocks.utils.config import Config

__all__ = [
    'import_by_name',
    'load_extra_args'
]


def import_by_name(name: str):
    last_dot_idx = name.rfind('.')
    if last_dot_idx < 0:
        raise RuntimeError(f'Failed to load {name}. Only module name is given.')
    module = import_module(name[:last_dot_idx])
    member = getattr(module, name[last_dot_idx + 1:])
    return member


def load_extra_args(config: Config) -> None:
    parser = argparse.ArgumentParser()
    for name, value in config.items():
        anno = config.get_type(name)

        kwargs = None
        if isinstance(anno, type):
            if issubclass(anno, (bool, str, int, float)):
                kwargs = {'type': anno}
            elif issubclass(anno, (tuple, list)):
                kwargs = {'nargs': '*'}
        elif hasattr(anno, '__origin__') and hasattr(anno, '__args__'):
            origin = anno.__origin__
            args = anno.__args__
            if origin == Literal and len(args) != 0:
                kwargs = {'type': type(args[0]), 'choices': args}
            elif origin in (tuple, list, collections.abc.Sequence):
                _type = args[0] if len(args) != 0 else None
                kwargs = {'type': _type, 'nargs': '*'}

        if kwargs:
            try:
                parser.add_argument(f'--{name}', **kwargs, default=value)
            except argparse.ArgumentError:
                pass

    config.load(parser.parse_known_args()[0])
