#!/usr/bin/env python3

import torch

from torchstocks.utils import fix_random_seed

fix_random_seed(42)


def main():
    a = torch.rand((10, 10))
    u, s, v = torch.svd_lowrank(a, 5)
    print(s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
