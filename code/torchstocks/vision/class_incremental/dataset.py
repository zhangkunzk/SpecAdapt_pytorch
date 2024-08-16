#!/usr/bin/env python3


from typing import Union, Sequence, Optional

from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.common.dataset import DSDataset, ImageTransform
from torchstocks.utils.image import RandomCrop

__all__ = [
    'TrainTransform',
    'TestTransform',
    'TrainDataset',
    'TestDataset'
]


class TrainTransform(ImageTransform):

    def __init__(
            self,
            image_size: int,
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            interpolation='area',
            augmenter: Optional[iaa.Augmenter] = None,
            image_field: str = 'image',
    ) -> None:
        super().__init__(
            image_field=image_field,
            augmenter=iaa.Sequential([
                augmenter,
                RandomCrop(
                    size=image_size,
                    shorter_side=shorter_side,
                    longer_side=longer_side,
                    interpolation=interpolation,
                    crop_position='uniform'
                )
            ])
        )


class TestTransform(ImageTransform):

    def __init__(
            self,
            image_size: int,
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            interpolation='area',
            image_field: str = 'image',
    ) -> None:
        super().__init__(
            image_field=image_field,
            augmenter=RandomCrop(
                size=image_size,
                shorter_side=shorter_side,
                longer_side=longer_side,
                interpolation=interpolation,
                crop_position='center'
            )
        )


class TrainDataset(Dataset):

    def __init__(
            self,
            path,
            image_size: int,
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            interpolation='area',
            augmenter: Optional[iaa.Augmenter] = None,
            image_field: str = 'image',
            rehearsal_dataset=None
    ) -> None:
        super().__init__()
        if rehearsal_dataset is None:
            self.dataset = DSDataset(path)
        else:
            self.dataset = rehearsal_dataset
        self.transform = TrainTransform(
            image_size=image_size,
            shorter_side=shorter_side,
            longer_side=longer_side,
            interpolation=interpolation,
            augmenter=augmenter,
            image_field=image_field
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


class TestDataset(Dataset):

    def __init__(
            self,
            path: Union[str, Sequence[str]],
            image_size: int,
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            interpolation='area',
            image_field: str = 'image',
    ) -> None:
        super().__init__()
        self.dataset = DSDataset(path)
        self.transform = TestTransform(
            image_size=image_size,
            shorter_side=shorter_side,
            longer_side=longer_side,
            interpolation=interpolation,
            image_field=image_field
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])
