#!/usr/bin/env python3

import random
import numpy as np
from tqdm import tqdm

import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.common.dataset import DataCollate, DSDataset, ImageDataset, DataTransform
from torchstocks.utils.bbox import array_to_bboxes_on_image, bboxes_on_image_to_array
from torchstocks.utils.image import normalize_image, IMAGENET_MEAN, Mosaic, read_image


__all__ = [
    'DetTrainDataset',
    'DetTestDataset',
    'DetTestSortedDataset',
    'DetDataCollate',
    'DetTestSortedDataCollate',
    'TestTransform'
]


FILL_VALUE = IMAGENET_MEAN.mean()
YOLO_FORMAT = True
RCNN_FORMAT = False
DATA_FORMAT = 'xywh'
PCT = True


def load_bboxes(objects):
    """Load bboxes
    """
    bboxes = np.empty((len(objects), 5), dtype=np.float32)
    for index, obj in enumerate(objects):
        x1, y1, x2, y2 = obj['box']
        label = obj['label']
        bboxes[index] = x1, y1, x2, y2, label
    return bboxes


class DetDataCollate(DataCollate):
    """Detection data collate
    """

    def __init__(
            self,
            image_field='image',
            bbox_field='bboxes'
    ) -> None:
        self.image_field = image_field
        self.bbox_field = bbox_field

    def __call__(self, doc_list):
        batch_doc = {}
        for doc in doc_list:
            for name, value in doc.items():
                if name not in batch_doc:
                    batch_doc[name] = []
                batch_doc[name].append(value)

        image_list = batch_doc[self.image_field]
        bboxes_list = batch_doc[self.bbox_field]

        for i in range(len(doc_list)):
            image_list[i] = torch.from_numpy(normalize_image(image_list[i], transpose=True))
            bboxes_list[i] = torch.from_numpy(bboxes_list[i])
        batch_doc[self.image_field] = torch.stack(image_list)
        return batch_doc


class DetTestSortedDataCollate(DataCollate):
    """Detection test sorted data collate
    """

    def __init__(
            self,
            image_field='image',
            bbox_field='bboxes',
            image_size=640,
            stride=32,
            fill_value=FILL_VALUE
    ) -> None:
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.image_size = image_size
        self.stride = stride
        self.fill_value = fill_value

    def __call__(self, doc_list):
        batch_doc = {}
        for doc in doc_list:
            for name, value in doc.items():
                if name not in batch_doc:
                    batch_doc[name] = []
                batch_doc[name].append(value)

        image_list = batch_doc[self.image_field]
        bboxes_list = batch_doc[self.bbox_field]

        r_list = []
        for image in image_list:
            h, w = image.shape[:2]
            r_list.append(w / h)
        r_min, r_max = min(r_list), max(r_list)
        if r_max < 1.0:
            wh = [r_max, 1.0]
        elif r_min > 1.0:
            wh = [1.0, 1.0 / r_min]
        else:
            wh = [1.0, 1.0]
        wh = np.ceil(np.array(wh) * self.image_size / self.stride + 0.5) * self.stride
        w, h = int(wh[0]), int(wh[1])

        pad = iaa.CenterPadToFixedSize(width=w, height=h, pad_cval=self.fill_value)
        for i in range(len(doc_list)):
            bboi = array_to_bboxes_on_image(bboxes_list[i], image_list[i].shape[0:2], 'xyxy', False)
            image_list[i], bboi = pad(image=image_list[i], bounding_boxes=bboi)
            bboxes_list[i] = bboxes_on_image_to_array(bboi, DATA_FORMAT, PCT)

        for i in range(len(doc_list)):
            image_list[i] = torch.from_numpy(normalize_image(image_list[i], transpose=True))
            bboxes_list[i] = torch.from_numpy(bboxes_list[i])
        batch_doc['image'] = torch.stack(image_list)
        return batch_doc


class DetTrainDataset(Dataset):
    """Detection train dataset
    """

    def __init__(
            self,
            path=None,
            image_field='image',
            bbox_field='bboxes',
            image_size=640,
            mosaic=True,
            p_scale=0.5,
            p_translate=0.1,
            p_flip_lr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fill_value=FILL_VALUE
    ) -> None:
        super(DetTrainDataset, self).__init__()

        self.image_field = image_field
        self.bbox_field = bbox_field
        self.mosaic = mosaic
        self.image_size = image_size
        assert YOLO_FORMAT != RCNN_FORMAT
        if mosaic:
            self.mosaic_aug = Mosaic(self.image_size, 0.0, cval=fill_value)

        geometry_transform = iaa.Sequential([
            iaa.Affine(
                scale=(1.0 - p_scale, 1.0 + p_scale),
                translate_percent={'x': (-p_translate, p_translate), 'y': (-p_translate, p_translate)},
                cval=fill_value
            ),
            iaa.Fliplr(p_flip_lr)
        ])
        size_transform = (
            iaa.CropToFixedSize(height=image_size, width=image_size) if self.mosaic else
            iaa.Sequential([
                iaa.Resize({'shorter-side': image_size, 'longer-side': 'keep-aspect-ratio'}),
                iaa.CropToFixedSize(height=image_size, width=image_size)
            ])
        )
        color_transform = iaa.WithColorspace(
            from_colorspace='RGB',
            to_colorspace='HSV',
            children=iaa.Sequential([
                iaa.WithChannels(0, iaa.Multiply((1.0 - hsv_h, 1.0 + hsv_h))),
                iaa.WithChannels(1, iaa.Multiply((1.0 - hsv_s, 1.0 + hsv_s))),
                iaa.WithChannels(2, iaa.Multiply((1.0 - hsv_v, 1.0 + hsv_v)))
            ])
        )
        self.transform = iaa.Sequential([
            geometry_transform,
            size_transform,
            color_transform,
        ])

        self.dataset = ImageDataset(
            DSDataset(path),
            image_field=image_field,
            bbox_field=bbox_field,
            augmenter=None,
            normalize=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
        image, objects = doc[self.image_field], doc[self.bbox_field]
        bboxes = load_bboxes(objects) # xyxy
        if self.mosaic:
            image_list, bboxes_list = [image], [bboxes]
            for _ in range(3):
                _doc = self.dataset[random.randint(0, len(self.dataset) - 1)]
                _image, _objects = _doc[self.image_field], _doc[self.bbox_field]
                _bboxes = load_bboxes(_objects)
                image_list.append(_image)
                bboxes_list.append(_bboxes)

            bboi_list = [
                array_to_bboxes_on_image(bboxes, image.shape[0:2], 'xyxy', False)
                for image, bboxes in zip(image_list, bboxes_list)
            ]
            image, bboi = self.mosaic_aug(images=image_list, bounding_boxes=bboi_list)
            bboxes = bboxes_on_image_to_array(bboi, 'xyxy', False)

        bboi = array_to_bboxes_on_image(bboxes, image.shape[0:2], 'xyxy', False) # xyxy
        image, bboi = self.transform(image=image, bounding_boxes=bboi) # xyxy
        bboxes = bboxes_on_image_to_array(bboi, DATA_FORMAT, PCT)  # xywh
        if YOLO_FORMAT:
            w, h = bboxes[..., 2], bboxes[..., 3]
        else:
            w = bboxes[..., 2] - bboxes[..., 0]
            h = bboxes[..., 3] - bboxes[..., 1]
        r = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        bboxes = bboxes[r < 100]

        doc = {
            'filename': doc['filename'],
            self.image_field: image,
            self.bbox_field: bboxes
        }
        return doc


class DetTestSortedDataset(Dataset):
    """Detection test sort dataset
    """

    def __init__(
            self,
            path=None,
            image_field='image',
            bbox_field='bboxes',
            image_size=640,
            stride=32,
            fill_value=FILL_VALUE
    ) -> None:
        super(DetTestSortedDataset, self).__init__()
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.docs = []
        self.dataset = ImageDataset(
            DSDataset(path),
            image_field=image_field,
            bbox_field=bbox_field,
            augmenter=None,
            normalize=False
        )
        for doc in tqdm(self.dataset, leave=False, ncols=96, desc='Loading test data'):
            image = doc[self.image_field]
            h, w = image.shape[:2]
            doc['ratio'] = w / h
            self.docs.append(doc)

        self.docs.sort(key=lambda d: d['ratio'])

        self.image_size = image_size
        self.stride = stride
        self.fill_value = fill_value

        self.transform = iaa.Resize(
            {'longer-side': self.image_size, 'shorter-side': 'keep-aspect-ratio'},
            interpolation='area'
        )

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]  # type: dict

        image, objects = doc[self.image_field], doc[self.bbox_field]
        bboxes = load_bboxes(objects)
        if max(image.shape[:2]) > self.image_size:
            bboi = array_to_bboxes_on_image(bboxes, image.shape[0:2], 'xyxy', False)
            image, bboi = self.transform(image=image, bounding_boxes=bboi)
            # bboxes = bboxes_on_image_to_array(bboi, DATA_FORMAT, PCT)
            bboxes = bboxes_on_image_to_array(bboi, 'xyxy', False)
        # print(bboxes)
        doc = {
            'filename': doc['filename'],
            self.image_field: image,
            self.bbox_field: bboxes
        }
        return doc


class TestTransform(DataTransform):
    """Test data transform
    """

    def __init__(
            self,
            image_size: int,
            image_field: str = 'image',
    ) -> None:
        self.image_field = image_field
        self.image_size = image_size
        self.transform = iaa.Sequential([
            iaa.Resize(
            {'longer-side': self.image_size, 'shorter-side': 'keep-aspect-ratio'},
            interpolation='area'),
            iaa.CenterPadToFixedSize(width=self.image_size, height=self.image_size, pad_cval=FILL_VALUE)
        ])

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


class DetTestDataset(Dataset):
    """Detection test dataset
    """

    def __init__(
            self,
            path=None,
            image_field='image',
            bbox_field='bboxes',
            image_size=640,
            stride=32,
            fill_value=FILL_VALUE
    ) -> None:
        super(DetTestDataset, self).__init__()
        self.image_field = image_field
        self.bbox_field = bbox_field
        assert YOLO_FORMAT != RCNN_FORMAT

        self.dataset = ImageDataset(
            DSDataset(path),
            image_field=image_field,
            bbox_field=bbox_field,
            augmenter=None,
            normalize=False
        )
        self.image_size = image_size
        self.fill_value = fill_value

        self.transform = iaa.Sequential([
            iaa.Resize(
            {'longer-side': self.image_size, 'shorter-side': 'keep-aspect-ratio'},
            interpolation='area'),
            iaa.CenterPadToFixedSize(width=self.image_size, height=self.image_size, pad_cval=self.fill_value)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]  # type: dict

        image, objects = doc[self.image_field], doc[self.bbox_field]
        bboxes = load_bboxes(objects)
        bboi = array_to_bboxes_on_image(bboxes, image.shape[0:2], 'xyxy', False)
        image, bboi = self.transform(image=image, bounding_boxes=bboi)
        bboxes = bboxes_on_image_to_array(bboi, DATA_FORMAT, PCT)
        doc = {
            'filename': doc['filename'],
            self.image_field: image,
            self.bbox_field: bboxes
        }
        return doc
