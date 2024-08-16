#!/usr/bin/env python3

"""
@author: Yubin
@since: 2022-10-31
"""

import random
from typing import Iterable, List, Sized, Literal

from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.common.dataset import DSDataset, ImageDataset, merge_docs


class OnlineClassIncrementalDataset(Dataset):

    def __init__(
            self,
            path_list: List[str],
            image_size: int,
            num_ways: int,
            num_shots: int,
            times_of_query: float,
            random_sample_num: int,
            train: bool = True,
            interpolation: Literal['nearest', 'linear', 'ares'] = 'linear'
    ) -> None:
        super().__init__()

        augmenter = iaa.Sequential([
            iaa.Resize((image_size, image_size), interpolation=interpolation)
        ])

        datasets = [
            ImageDataset(
                DSDataset(path),
                augmenter=augmenter
            )
            for path in path_list
        ]
        self.dataset = OnlineCWayKShotDataset(
            datasets=datasets,
            num_ways=num_ways,
            num_shots=num_shots,
            times_of_query=times_of_query,
            random_sample_num=random_sample_num,
            train=train,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OnlineCWayKShotDataset(Dataset):

    def __init__(
            self,
            datasets: Iterable[Dataset],
            num_ways: int,
            num_shots: int,
            times_of_query: float = 0.,
            random_sample_num: int = 64,
            train=True,
            merge_fn=merge_docs
    ) -> None:
        super(OnlineCWayKShotDataset, self).__init__()
        self.datasets = [*datasets]
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.times_of_query = times_of_query
        self.random_sample_num = random_sample_num if train else 0
        self.train = train
        self.merge_fn = merge_fn

        self.tasks = []
        self.all_sample_idx = []
        for c, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Sized)
            self.tasks.append((c, []))
            for sample_idx in range(len(dataset)):
                self.tasks[c][1].append(sample_idx)
                self.all_sample_idx.append((c, sample_idx))
        self.size = sum(len(sample[1]) for sample in self.tasks)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        query_docs = []
        supp_docs = []
        sample_tasks = random.sample(self.tasks, self.num_ways)
        sample_image_num = int(self.num_shots * (1 + self.times_of_query))

        for dataset_idx, index_list in sample_tasks:
            dataset = self.datasets[dataset_idx]
            sample_index = random.sample(index_list, sample_image_num)
            for supp_idx in sample_index[:self.num_shots]:
                supp_doc = dataset[supp_idx]
                supp_docs.append(supp_doc)
            if self.times_of_query == 0:
                query_docs.extend(supp_docs)
            else:
                for query_idx in sample_index[self.num_shots:]:
                    query_doc = dataset[query_idx]
                    query_docs.append(query_doc)

        if self.random_sample_num > 0:
            part = random.sample(self.all_sample_idx, self.random_sample_num)
            for dataset_idx, sample_idx in part:
                query_doc = self.datasets[dataset_idx][sample_idx]
                query_docs.append(query_doc)

        if callable(self.merge_fn):
            query_docs = self.merge_fn(query_docs)
            supp_docs = self.merge_fn(supp_docs)

        return query_docs, supp_docs
