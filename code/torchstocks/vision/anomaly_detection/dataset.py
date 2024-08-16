#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-09-15
"""

import os

import cv2 as cv
import numpy as np
from docset import DocSet
from imgaug import augmenters as iaa, SegmentationMapsOnImage
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def encode_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return image


def decode_image(tensor: np.ndarray) -> np.ndarray:
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = tensor * STD + MEAN
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)


def encode_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.array(mask, np.float32)
    mask = np.transpose(mask, (2, 0, 1))
    return mask


def decode_mask(tensor: np.ndarray) -> np.ndarray:
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.clip(tensor, 0, 255)
    tensor = np.array(tensor, np.uint8)
    return tensor


class CLAHE(iaa.Augmenter):

    def get_parameters(self):
        pass

    def __init__(self, clip_limit=1.0, grid_size=(2, 2)):
        super(CLAHE, self).__init__()
        self.impl = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def __call__(self, image):
        img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = self.impl.apply(img_yuv[:, :, 0])
        return cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)


class ImageTransform(object):

    def __init__(self, iaa_fn, hist_enhance: bool):
        self._iaa_fn = iaa_fn
        self.enhance = CLAHE() if hist_enhance else None

    def __call__(self, image, mask=None):
        if self.enhance is not None:
            image = self.enhance(image)
        if mask is None:
            image = self._iaa_fn(image=image)
        else:
            seg_maps = SegmentationMapsOnImage(mask, shape=mask.shape)
            image, seg_maps = self._iaa_fn(image=image, segmentation_maps=seg_maps)
            mask = seg_maps.arr
        return image, mask


class TestTransform(ImageTransform):

    def __init__(self, height: int, width: int, hist_enhance: bool):
        super(TestTransform, self).__init__(
            iaa.Sequential([
                iaa.Resize({'height': height, 'width': width}, interpolation='area')
            ]) if (height and width) else iaa.Identity(),
            hist_enhance
        )


class TrainTransform(ImageTransform):

    def __init__(self, height: int, width: int, hist_enhance: bool):
        resize = iaa.Sequential([
            iaa.Resize(
                {'height': (height, int(height * 1.05)), 'width': (width, int(width * 1.05))},
                interpolation='area'
            ),
            iaa.CropToFixedSize(height, width)
        ]) if (height and width) else iaa.Identity()
        super(TrainTransform, self).__init__(
            iaa.Sequential([
                iaa.AddToBrightness((-5, 5)),
                resize
            ]),
            hist_enhance
        )


class ADDataset(Dataset):

    def __init__(self, ds_path, transform=None):
        self._ds = DocSet(ds_path, 'r')
        self._transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        doc = self._ds[i]
        image = cv.imdecode(np.frombuffer(doc['image'], np.byte), cv.IMREAD_COLOR)
        image = np.flip(image, -1)
        if 'label' in doc:
            mask = cv.imdecode(np.frombuffer(doc['label'], np.byte), cv.IMREAD_GRAYSCALE)
            mask = mask[:, :, None]
        else:
            mask = None

        if callable(self._transform):
            image, mask = self._transform(image, mask)

        image = encode_image(image)
        if mask is not None:
            mask = encode_mask(mask)
        else:
            _, h, w = image.shape
            mask = np.zeros((1, h, w), np.float32)

        ab_label = 0 if doc['defect'] == 'good' else 1

        return {
            'image': image,
            'mask': mask,
            'label': ab_label
        }


class DirDataset(Dataset):

    def __init__(self, path: str, good_label: str = 'good', only_good=False, transform=None):
        super(DirDataset, self).__init__()
        assert os.path.isdir(path)
        self.path = path
        self.good_label = good_label
        self.transform = transform

        self.docs = []
        for dir_name in os.listdir(path):
            if only_good and dir_name != good_label:
                continue
            dir_path = os.path.join(path, dir_name)
            for filename in os.listdir(dir_path):
                doc = {
                    'image_path': os.path.join(dir_path, filename),
                    'label': 0 if dir_name == self.good_label else 1
                }
                self.docs.append(doc)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        doc = self.docs[i]
        image = cv.imread(doc['image_path'], cv.IMREAD_COLOR)
        image = np.flip(image, -1)

        if callable(self.transform):
            image, mask = self.transform(image)
        image = encode_image(image)

        # _, h, w = image.shape
        # mask = np.zeros((1, h, w), np.float32)

        return {
            'image': image,
            # 'mask': mask,
            'label': doc['label']
        }
