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
    """Train dataset transform
    """

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
                    crop_position='uniform',
                    interpolation=interpolation
                )
            ])
        )


class TestTransform(ImageTransform):
    """Test dataset transform
    """

    def __init__(
            self,
            image_size: int,
            interpolation='area',
            image_field: str = 'image',
    ) -> None:
        super().__init__(
            image_field=image_field,
            augmenter=RandomCrop(
                size=image_size,
                crop_position='center',
                interpolation=interpolation
            )
        )


class TrainDataset(Dataset):
    """Train dataset
    """

    def __init__(
            self,
            path: Union[str, Sequence[str]],
            image_size: int,
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            interpolation='area',
            augmenter: Optional[iaa.Augmenter] = None,
            image_field: str = 'image',
    ) -> None:
        super().__init__()
        self.dataset = DSDataset(path)
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
    """Test dataset
    """

    def __init__(
            self,
            path: Union[str, Sequence[str]],
            image_size: int,
            interpolation='area',
            image_field: str = 'image',
    ) -> None:
        super().__init__()
        self.dataset = DSDataset(path)
        self.transform = TestTransform(
            image_size=image_size,
            interpolation=interpolation,
            image_field=image_field
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])
