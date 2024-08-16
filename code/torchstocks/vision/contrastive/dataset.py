#!/usr/bin/env python3

from typing import Union, Sequence, Optional

import imgaug.augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.common.dataset import DSDataset
from torchstocks.utils.image import normalize_image, RandomCrop, read_image


class UnsupervisedDataset(Dataset):

    def __init__(
            self,
            path: Union[str, Sequence[str]],
            image_size: int,
            shorter_side: Union[int, float] = None,
            longer_side: Union[int, float] = None,
            augmenter: Optional[iaa.Augmenter] = None,
            image_field: str = 'image'
    ) -> None:
        super(UnsupervisedDataset, self).__init__()

        if isinstance(shorter_side, float):
            shorter_side = int(shorter_side * image_size)
        if isinstance(longer_side, float):
            longer_side = int(longer_side * image_size)
        if augmenter is None:
            augmenter = iaa.Identity()

        self.dataset = DSDataset(path)
        self.transform = iaa.Sequential([
            augmenter,
            RandomCrop(
                size=image_size,
                shorter_side=shorter_side,
                longer_side=longer_side,
                crop_position='uniform'
            )
        ])
        self.image_field = image_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
        image = read_image(doc[self.image_field])
        image1 = self.transform(image=image)
        image2 = self.transform(image=image)

        image1 = normalize_image(image1, transpose=True)
        image2 = normalize_image(image2, transpose=True)
        doc[self.image_field] = (image1, image2)
        return doc


class SupervisedDataset(Dataset):

    def __init__(
            self,
            path: Union[str, Sequence[str]],
            image_size: int,
            image_field='image'
    ) -> None:
        super(SupervisedDataset, self).__init__()
        self.dataset = DSDataset(path)
        # self.transform = iaa.Resize({'height': image_size, 'width': image_size})
        self.transform = RandomCrop(
            size=image_size,
            shorter_side=image_size,
            crop_position='center'
        )
        self.image_field = image_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
        image = read_image(doc[self.image_field])
        image = self.transform(image=image)
        image = normalize_image(image, transpose=True)
        doc[self.image_field] = image
        return doc
