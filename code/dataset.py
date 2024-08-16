# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


def flist_reader(flist):
    imlist = []
    with open(flist, 'r') as file:
        for line in file.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(Dataset):

    def __init__(self, root, flist, transform=None, target_transform=None):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        image = Image.open(os.path.join(self.root, impath)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'image': image, 'label': target}

    def __len__(self):
        return len(self.imlist)


def get_dataset(
    name,
    image_size=224,
    evaluate=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    root='../../local_data/vtab-1k/' + name
    trainval_flist=root + "/train800val200.txt"
    train_flist=root +"/train800.txt"
    val_flist=root + "/val200.txt"
    test_flist=root + "/test.txt"
    train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
    if evaluate:
        train_set = ImageFilelist(root=root, flist=trainval_flist, transform=train_transform)
        test_set = ImageFilelist(root=root, flist=test_flist, transform=val_transform)
    else:
        train_set = ImageFilelist(root=root, flist=train_flist, transform=train_transform)
        test_set = ImageFilelist(root=root, flist=val_flist, transform=val_transform)

    return train_set, test_set
