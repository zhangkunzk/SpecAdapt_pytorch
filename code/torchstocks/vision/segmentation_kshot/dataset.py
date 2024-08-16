#!/usr/bin/env python3


from typing import List, Optional, Union

from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.common.dataset import DSDataset, ImageDataset, KShotDataset
from torchstocks.utils.image import RandomCrop

IGNORE_CLASS = 255


class KShotSegmentationDataset(Dataset):

    def __init__(
            self,
            path_list: List[str],  # e.g., ['task_01.ds', 'task_02.ds', 'tasK_03.ds']
            num_shots: int,  # e.g., 5
            image_size: int,  # e.g., 473
            shorter_side: Optional[Union[int, float]] = None,
            longer_side: Optional[Union[int, float]] = None,
            augmenter: Optional[iaa.Augmenter] = None,
            train: bool = False,
    ) -> None:
        super(KShotSegmentationDataset, self).__init__()

        # s = image_size
        # augmenter = iaa.Sequential([
        #     iaa.Fliplr(0.5),
        #     ColorJitter(),
        #
        #     iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center'),
        #     iaa.Resize({'longer-side': (s, int(s * 1.2)), 'shorter-side': 'keep-aspect-ratio'}, interpolation),
        #     iaa.CropToFixedSize(s, s),
        #
        #     iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, cval=127),
        #     iaa.Rotate((-5, 5), cval=127),
        # ]) if train else iaa.Sequential([
        #     iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation),
        #     iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center')
        # ])

        def _record_image_size(doc):
            image = doc['image']
            doc['size'] = max(image.shape[0], image.shape[1])

        datasets = [
            ImageDataset(
                DSDataset(path),
                mask_field='mask',
                pre_augment=_record_image_size,
                augmenter=iaa.Sequential([
                    augmenter,
                    RandomCrop(
                        size=image_size,
                        shorter_side=shorter_side,
                        longer_side=longer_side,
                        crop_position='uniform' if train else 'center',
                    )
                ])
            )
            for path in path_list
        ]
        self.dataset = KShotDataset(
            datasets,
            num_shots=num_shots,
            no_overlap=not train
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
