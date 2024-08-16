#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-11-11
"""

from typing import List

from imgaug import augmenters as iaa

from torchstocks.common.dataset import DSDataset, ImageDataset, CWayKShotDataset


class FewShotDateset(CWayKShotDataset):

    def __init__(
            self,
            path_list: List[str],  # [class1.ds, class2.ds, class3.ds, ...]
            num_ways: int,
            num_shots: int,
            image_size: int,
            train: bool,
            no_overlop: bool = True,
            rewrite_label=True,
    ) -> None:

        if train:
            augmenter = iaa.Sequential([
                iaa.Resize((image_size, image_size), interpolation='linear'),
                iaa.Fliplr(0.5)
            ])
        else:
            augmenter = iaa.Resize((image_size, image_size), interpolation='linear')

        datasets = [
            ImageDataset(
                DSDataset(path),
                augmenter=augmenter
            )
            for path in path_list
        ]

        super().__init__(
            datasets=datasets,
            num_ways=num_ways,
            num_shots=num_shots,
            no_overlap=no_overlop,
            rewrite_label=rewrite_label,
        )
