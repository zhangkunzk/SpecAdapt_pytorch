#!/usr/bin/env python3

import collections
import glob
import os
import random
from typing import Iterable, Mapping, Callable, MutableMapping, Sized
from typing import Union, Sequence

import numpy as np
import torch
from docset import DocSet, ConcatDocSet
from imgaug import SegmentationMapsOnImage, BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.utils.image import read_image, normalize_image

__all__ = [
    'DSDataset',
    'ImageDataset',
    'merge_docs',
    'KShotDataset',
    'CWayKShotDataset'
]


class DSDataset(Dataset):

    def __init__(self, path: Union[str, Sequence[str]]):
        path_list = []
        if isinstance(path, str):
            if os.path.isdir(path):
                path_list.extend(glob.iglob(os.path.join(path, '*.ds')))
            else:
                path_list.append(path)
        else:
            path_list.extend(path)

        self.ds_list = [DocSet(path) for path in path_list]
        self.ds = self.ds_list[0] if len(self.ds_list) == 1 else ConcatDocSet(self.ds_list)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class ImageDataset(Dataset):

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
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.image_field = image_field
        self.bbox_field = bbox_field
        self.mask_field = mask_field
        self.pre_augment = pre_augment
        self.augmenter = augmenter
        self.post_augment = post_augment
        self.normalize = normalize
        self.transpose = transpose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset[idx]
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


def merge_docs(doc_list):
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
