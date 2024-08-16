#!/usr/bin/env python3

import abc
import collections
import glob
import os
import random
from typing import Iterable, Mapping, Callable, MutableMapping, Sized
from typing import Union, Sequence

import cv2 as cv
import numpy as np
import torch
from docset import DocSet, ConcatDocSet
from imgaug import SegmentationMapsOnImage, BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.utils.image import read_image, normalize_image

__all__ = [
    'DataTransform',
    'DataCollate',
    'DSDataset',
    'ImageTransform',
    'ImageDataset',
    'merge_docs',
    'KShotDataset',
    'CWayKShotDataset'
]


class DSDataset(Dataset):
    """Read docset format dataset
    """

    def __init__(self, path: Union[str, Sequence[str]]):
        super().__init__()
        path_list = []
        if path is not None:
            if isinstance(path, str):
                if os.path.isdir(path):
                    path_list.extend(glob.iglob(os.path.join(path, '*.ds')))
                else:
                    path_list.append(path)
            else:
                path_list.extend(path)

        self.ds_list = [DocSet(path) for path in path_list]
        if len(self.ds_list) == 1:
            self.ds = self.ds_list[0]
        elif len(self.ds_list) > 1:
            self.ds = ConcatDocSet(self.ds_list)
        else:
            self.ds = []

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class DataTransform(abc.ABC):
    """Data transform
    """

    @abc.abstractmethod
    def __call__(self, sample):
        pass


class DataCollate(abc.ABC):
    """Data collate
    """

    @abc.abstractmethod
    def __call__(self, samples: Sequence):
        pass


class ImageTransform(DataTransform):
    """Image transform
    """

    def __init__(
            self,
            image_field='image',
            bbox_field=None,
            mask_field=None,
            pre_augment: Callable[[MutableMapping], None] = None,
            augmenter: iaa.Augmenter = None,
            post_augment: Callable[[MutableMapping], None] = None,
            normalize=True,
            transpose=True
    ) -> None:
        super(ImageTransform, self).__init__()
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.mask_field = mask_field
        self.pre_augment = pre_augment
        self.augmenter = augmenter
        self.post_augment = post_augment
        self.normalize = normalize
        self.transpose = transpose

    def __call__(self, doc):
        doc[self.image_field] = read_image(doc[self.image_field])

        if callable(self.pre_augment):
            self.pre_augment(doc)
        self._apply_augmenter(doc)
        if callable(self.post_augment):
            self.post_augment(doc)

        if self.normalize:
            doc[self.image_field] = normalize_image(doc[self.image_field], transpose=self.transpose)
        return doc

    def _apply_augmenter(self, doc):
        if self.augmenter is None:
            return

        if not isinstance(self.augmenter, iaa.Augmenter):
            raise RuntimeError('Invalid augmenter.')

        ################################################################################
        # prepare augmenter arguments
        ################################################################################
        aug_args = {'image': doc[self.image_field]}

        if self.bbox_field:
            image = doc[self.image_field]
            bboxes = doc[self.bbox_field]
            bbox_objs = []
            for bbox in bboxes:
                # x, y, w, h, label = bbox
                # x, y, w, h = x * iw, y * ih, w * iw, h * ih
                # ow, oh = w * 0.5, h * 0.5
                # x1, y1, x2, y2 = x - ow, y - oh, x + ow, y + oh
                x1, y1, x2, y2, label = bbox
                bbox_obj = BoundingBox(x1, y1, x2, y2, label)
                bbox_objs.append(bbox_obj)
            aug_args['bounding_boxes'] = BoundingBoxesOnImage(
                bounding_boxes=bbox_objs,
                shape=image.shape
            )

        if self.mask_field:
            image = doc[self.image_field]
            mask = doc[self.mask_field]
            if isinstance(mask, bytes):
                mask = cv.imdecode(np.frombuffer(mask, np.byte), cv.IMREAD_GRAYSCALE)
            assert isinstance(mask, np.ndarray)
            mask_rank = len(mask.shape)
            assert mask_rank == 2 or mask_rank == 3
            aug_args['segmentation_maps'] = SegmentationMapsOnImage(
                arr=mask,
                shape=image.shape
            )

        ################################################################################
        # perform augmentation
        ################################################################################
        aug_result = self.augmenter(**aug_args)
        if len(aug_args) == 1:
            aug_result = [aug_result]

        ################################################################################
        # collect augmentation results
        ################################################################################
        aug_result = iter(aug_result)
        doc[self.image_field] = next(aug_result)

        if self.bbox_field:
            bbox_objs = next(aug_result)
            bbox_objs = bbox_objs.remove_out_of_image_fraction(0.8).clip_out_of_image()
            bboxes = np.empty((len(bbox_objs), 5), dtype=np.float32)
            for i, bbox_obj in enumerate(bbox_objs):
                x1, y1, x2, y2, label = bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2, bbox_obj.label
                # x, y, w, h = (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1
                # x, y, w, h = x / iw, y / ih, w / iw, h / ih
                # bboxes[i] = x, y, w, h, label
                bboxes[i] = x1, y1, x2, y2, label
            doc[self.bbox_field] = bboxes

        if self.mask_field:
            maps_oi = next(aug_result)
            mask = maps_oi.arr.squeeze(-1)
            doc[self.mask_field] = np.array(mask, np.int64)


class ImageDataset(Dataset):
    """Image dataset
    """

    def __init__(
            self,
            dataset,
            image_field='image',
            bbox_field=None,
            mask_field=None,
            pre_augment: Callable[[MutableMapping], None] = None,
            augmenter: iaa.Augmenter = None,
            post_augment: Callable[[MutableMapping], None] = None,
            normalize=True,
            transpose=True
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = ImageTransform(
            image_field=image_field,
            bbox_field=bbox_field,
            mask_field=mask_field,
            pre_augment=pre_augment,
            augmenter=augmenter,
            post_augment=post_augment,
            normalize=normalize,
            transpose=transpose
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def merge_docs(doc_list):
    """Merge docs
    """
    if isinstance(doc_list[0], Mapping):
        merged = collections.defaultdict(list)
        for doc in doc_list:
            for name, value in doc.items():
                if isinstance(value, int):
                    value = np.array(value, np.int64)
                merged[name].append(value)
        result = {}
        for name, values in merged.items():
            try:
                if isinstance(values[0], np.ndarray):
                    values = np.stack(values)
                elif isinstance(values[0], torch.Tensor):
                    values = torch.stack(values)
            except ValueError:
                pass
            result[name] = values
        return result
    else:
        return doc_list


class KShotDataset(Dataset):
    """K-shot dataset
    """

    def __init__(
            self,
            datasets: Iterable[Dataset],
            num_shots: int,
            no_overlap=False,
            merge_fn=merge_docs
    ) -> None:
        super(KShotDataset, self).__init__()
        self.datasets = [*datasets]
        self.num_shots = num_shots
        self.no_overlap = no_overlap
        self.merge_fn = merge_fn

        self.query_list = []
        self.supp_dict = collections.defaultdict(list)
        for dataset_idx, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Sized)
            for sample_idx in range(len(dataset)):
                if self.no_overlap:
                    if sample_idx % 2 == 0:
                        self.query_list.append((dataset_idx, sample_idx))
                    else:
                        self.supp_dict[dataset_idx].append(sample_idx)
                else:
                    self.query_list.append((dataset_idx, sample_idx))
                    self.supp_dict[dataset_idx].append(sample_idx)

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        dataset_idx, query_idx = self.query_list[idx]
        supp_list = self.supp_dict[dataset_idx]
        supp_idx_list = random.sample(supp_list, self.num_shots)

        dataset = self.datasets[dataset_idx]
        query_doc = dataset[query_idx]
        supp_docs = [dataset[supp_idx] for supp_idx in supp_idx_list]

        if callable(self.merge_fn):
            supp_docs = self.merge_fn(supp_docs)

        return query_doc, supp_docs


class CWayKShotDataset(Dataset):
    """C-way k-shot dataset
    """

    def __init__(
            self,
            datasets: Iterable[Dataset],
            num_ways: int,
            num_shots: int,
            no_overlap=False,
            rewrite_label=True,
            label_field='label',
            merge_fn=merge_docs
    ) -> None:
        super(CWayKShotDataset, self).__init__()
        self.datasets = [*datasets]
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.no_overlap = no_overlap
        self.rewrite_label = rewrite_label
        self.label_field = label_field
        self.merge_fn = merge_fn

        self.partitions = [(dataset, [], []) for dataset in self.datasets]
        for c, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Sized)
            for sample_idx in range(len(dataset)):
                if self.no_overlap:
                    self.partitions[c][(sample_idx % 2) + 1].append(sample_idx)
                else:
                    self.partitions[c][1].append(sample_idx)
                    self.partitions[c][2].append(sample_idx)
        self.size = sum(len(sample[1]) for sample in self.partitions)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        query_docs = []
        supp_docs = []
        sub_partition = random.sample(self.partitions, self.num_ways)
        for label, (dataset, query_list, supp_list) in enumerate(sub_partition):
            for query_idx in random.sample(query_list, self.num_shots):
                query_doc = dataset[query_idx]
                if self.rewrite_label:
                    query_doc[self.label_field] = label
                query_docs.append(query_doc)
            for supp_idx in random.sample(supp_list, self.num_shots):
                supp_doc = dataset[supp_idx]
                if self.rewrite_label:
                    supp_doc[self.label_field] = label
                supp_docs.append(supp_doc)

        if callable(self.merge_fn):
            query_docs = self.merge_fn(query_docs)
        if callable(self.merge_fn):
            supp_docs = self.merge_fn(supp_docs)

        return query_docs, supp_docs


class RehearsalDSDataset(Dataset):
    """Rehearsal docset dataset
    """

    def __init__(self,
                 path: Union[str, Sequence[str]],
                 write_path: str,
                 label_field: str = 'label',
                 samples_pre_class: int = 20,
                 fix_memory_size: int = 2000
                 ):

        self.label_field = label_field
        self.samples_pre_class = samples_pre_class
        self.write_path = write_path
        self.fix_memory_size = fix_memory_size
        if os.path.exists(path):
            self.ds = DocSet(path)
        else:
            self.ds = None

    def __len__(self):
        if self.ds:
            return len(self.ds)
        else:
            return 0

    def __getitem__(self, i):
        return self.ds[i]

    def write(self, dataset, num_class):
        """Write docset data
        """
        sampling_idx = [[] for _ in range(num_class)]
        for idx, doc in enumerate(dataset):
            class_ = doc[self.label_field]
            if len(sampling_idx[class_]) < self.samples_pre_class:
                sampling_idx[class_].append(idx)
            else:
                random_position = random.randint(0, self.samples_pre_class - 1)
                sampling_idx[class_][random_position] = idx
        with DocSet(self.write_path, 'w') as ds:
            for _, idx_list_i in enumerate(sampling_idx):
                for j in idx_list_i:
                    ds.write(dataset[j])
            if self.ds is not None:
                for doc_i in self.ds:
                    ds.write(doc_i)

    def write_fixed_size(self, dataset, num_class):
        """
        rehearsal dataset has a fixed size
        """
        buffer = []
        if self.ds is not None:
            for doc_i in self.ds:
                buffer.append(doc_i)
        for idx, doc in enumerate(dataset):
            buffer.append(doc)
        sample_pre_class = self.fix_memory_size // num_class
        sampling_idx = [[] for _ in range(num_class)]
        for idx, doc in enumerate(buffer):
            class_ = doc[self.label_field]
            if len(sampling_idx[class_]) < sample_pre_class:
                sampling_idx[class_].append(idx)
            else:
                random_position = random.randint(0, self.samples_pre_class - 1)
                sampling_idx[class_][random_position] = idx
        with DocSet(self.write_path, 'w') as ds:
            for _, idx_list_i in enumerate(sampling_idx):
                for j in idx_list_i:
                    ds.write(buffer[j])
