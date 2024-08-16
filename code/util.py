#!/usr/bin/env python3


from model.vit import Model

from torchstocks.optim.lorsa.common import is_target_group, FlatLikeSquare
from torchstocks.optim.lorsa.svd import SVDDecomposition, SVDDecompositionV1
from torchstocks.optim.lorsa.freq import FreqDecomposition
from torchstocks.optim.lorsa.xxx import Decomposition, create_dct_matrix
from torchstocks.common.trainer import parse_param_groups


_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
_CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)


def get_classes_num(dataset_name):
    dict_ = {name: num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}
    return dict_[dataset_name]


def get_mh_mw(params):
    dtype, device = None, None
    max_h, max_w = 0, 0
    for p in params:
        if len(p.shape) < 2:
            continue
        h, w = FlatLikeSquare(p.shape).target_shape[-2:]
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
    if dtype is None or device is None:
        dtype = p.dtype
        device = p.device
    if max_h > 0 and max_w > 0:
        mh = create_dct_matrix(max_h, dtype=dtype, device=device)
        mw = create_dct_matrix(max_w, dtype=dtype, device=device)
    return mh, mw


def get_trainable_params_num(model, params_group, r, flag=''):
    total = 0
    params = parse_param_groups(params_group, model, verbose=True)
    for group in params:
        if not is_target_group(group):
            continue
        params = group['params']
        if flag == 'xxx':
            mh, mw = get_mh_mw(params)
        for p in params:
            try:
                if flag == 'svd':
                    decomposition = SVDDecomposition(p, r)
                elif flag == 'xxx':
                    decomposition = Decomposition(p, r, mh, mw)
                    decomposition.init()
                else:
                    decomposition = FreqDecomposition(p, r)

                for z in decomposition.params:
                    print(f'need to update: {z.shape}')
                    total += z.numel()
            except ValueError:
                continue
        if flag == 'xxx':
            total += mh.numel()
            total += mw.numel()
    print(f'Number of trainable params: {(total) / 1e6} M')


if __name__ == '__main__':
    params_group = [{"match": ["^.*(blocks).*(attn.(qkv|proj).*weight)$", "^.*mlp.*weight$"], "lr": 1e-4, "tag": "low_rank"}]
    get_trainable_params_num(Model(100), params_group, r=8, flag='xxx')