#!/usr/bin/env python3
"""
@author: liying50
@since: 2022-05-07
"""

from typing import Sequence

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from torchstocks.common.dataset import DSDataset, ImageDataset, DataCollate, DataTransform
from torchstocks.utils.image import ColorJitter, normalize_image, read_image

__all__ = [
    'SegTrainDataset',
    'SegTestDataset',
    'SegDataCollate'
]


def encode_mask(mask: np.ndarray) -> np.ndarray:
    """Encode mask
    """
    mask = np.array(mask, dtype=np.int64)
    return mask


class SegDataCollate(DataCollate):
    """Segmentation data collate
    """

    def __init__(
            self,
            image_field='image',
            mask_field='mask'
    ) -> None:
        self.image_field = image_field
        self.mask_field = mask_field

    def __call__(self, doc_list):
        batch_doc = {}
        for doc in doc_list:
            for name, value in doc.items():
                if name not in batch_doc:
                    batch_doc[name] = []
                batch_doc[name].append(value)

        image_list = batch_doc[self.image_field]
        label_list = batch_doc[self.mask_field]

        for i in range(len(doc_list)):
            image_list[i] = torch.from_numpy(
                normalize_image(image_list[i], transpose=True))
            label_list[i] = torch.from_numpy(encode_mask(label_list[i]))
        batch_doc[self.image_field] = torch.stack(image_list)
        batch_doc[self.mask_field] = torch.stack(label_list)
        return batch_doc


class SegTrainDataset(Dataset):
    """Segmentation train dataset
    """

    def __init__(
            self,
            path: str,
            image_field='image',
            mask_field='mask',
            image_size=513,
            p_flip_lr=0.5,
            p_scale=0.5,
            rnd_rotate=30

    ) -> None:
        super(SegTrainDataset, self).__init__()
        self.image_field = image_field
        self.mask_field = mask_field
        interpolation = 'linear'
        if isinstance(image_size, Sequence):
            s = max(image_size)
        else:
            s = image_size
            image_size = (image_size, image_size)
        self.transform = iaa.Sequential([
            iaa.Fliplr(p_flip_lr),
            ColorJitter(),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center'),
            iaa.Resize({'longer-side': (s, int(s * 1.2)),
                       'shorter-side': 'keep-aspect-ratio'}, interpolation),
            iaa.CropToFixedSize(height=image_size[0], width=image_size[1]),
            iaa.Affine(scale=(p_scale, 1 / p_scale), cval=127),
            iaa.Rotate((-rnd_rotate, rnd_rotate), cval=127),
        ])

        self.dataset = ImageDataset(
            DSDataset(path),
            image_field=image_field,
            mask_field=mask_field,
            augmenter=None,
            normalize=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
        image, mask = doc[self.image_field], doc[self.mask_field]
        mask = cv.imdecode(np.frombuffer(mask, np.byte), cv.IMREAD_GRAYSCALE)
        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, mask = self.transform(image=image, segmentation_maps=mask)
        doc = {
            'filename': doc['filename'],
            self.image_field: image,
            self.mask_field: mask.arr.squeeze(2)
        }
        return doc


class SegTestDataset(Dataset):
    """Segmentation test dataset
    """

    def __init__(
            self,
            path: str,
            image_field='image',
            mask_field='mask',
            image_size=513
    ) -> None:
        super(SegTestDataset, self).__init__()
        self.image_field = image_field
        self.mask_field = mask_field
        interpolation = 'linear'
        if isinstance(image_size, Sequence):
            image_size = max(image_size)
        self.transform = iaa.Sequential([
            iaa.Resize({'longer-side': image_size,
                       'shorter-side': 'keep-aspect-ratio'}, interpolation),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center')
        ])
        self.dataset = ImageDataset(
            DSDataset(path),
            image_field=image_field,
            mask_field=mask_field,
            augmenter=None,
            normalize=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
        image, mask = doc[self.image_field], doc[self.mask_field]
        mask = cv.imdecode(np.frombuffer(mask, np.byte), cv.IMREAD_GRAYSCALE)
        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, mask = self.transform(image=image, segmentation_maps=mask)
        doc = {
            'filename': doc['filename'],
            self.image_field: image,
            self.mask_field: mask.arr.squeeze(2)
        }
        return doc


class TestTransform(DataTransform):
    """Test dataset transform
    """

    def __init__(
            self,
            image_size,
            image_field='image',
    ) -> None:
        self.image_field = image_field
        interpolation = 'linear'
        self.transform = iaa.Resize(
            {'longer-side': image_size, 'shorter-side': 'keep-aspect-ratio'}, interpolation)

    def __call__(self, doc):
        image_buffer = doc[self.image_field]
        image = read_image(image_buffer)
        image = self.transform(image=image)
        doc[self.image_field] = normalize_image(image, transpose=True)
        return doc


class InferenceTransform(DataTransform):
    """Transform an RGB image directly
    """

    def __init__(
            self,
            image_size: int
    ) -> None:
        self.image_size = image_size
        self.transform = iaa.Resize(
            {'longer-side': self.image_size, 'shorter-side': 'keep-aspect-ratio'},
            interpolation='linear'
        )

    def __call__(self, image):
        resized_image = self.transform(image=image)
        return normalize_image(resized_image, transpose=True)
