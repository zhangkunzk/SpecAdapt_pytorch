#!/usr/bin/env python3

from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.new_common.tester import AbstractTester, init_test_dataset
from torchstocks.utils.metrics import IouMeter
from torchstocks.utils.image import normalize_image
from torchstocks.common.dataset import DataCollate


__all__ = [
    'SegmentationTester'
]


class SegmentationTester(AbstractTester):
    """Segmentation tester
    """
    def __init__(
            self,
            model: nn.Module,
            test_dataset: Dataset,
            test_collate: DataCollate,
            batch_size: int = 32,
            num_workers: int = 10,
            input_field: str = 'image',
            target_field: str = 'mask',
            device: str = 'cpu',
            output_dir: str = None
    ) -> None:
        super(SegmentationTester, self).__init__()
        self.model = model
        self.model.eval()
        self.input_field = input_field
        self.target_field = target_field
        self.device = device
        self.output_dir = output_dir
        self.test_dataset, self.test_loader, self.test_transform = init_test_dataset(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=test_collate
        )

    def _inference(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(inputs=image, targets=None)
            return outputs

    def __call__(self, doc):
        if callable(self.test_transform):
            doc = self.test_transform(image=doc)
        x = normalize_image(doc, transpose=True)
        x = torch.as_tensor(x).unsqueeze(0)
        outputs = self._inference(x)
        return outputs[0]

    def run(self):
        """Run
        """
        if self.test_loader is None:
            return

        meter = IouMeter(ignore_class=255, bg_class=-1)
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            image, targets = doc[self.input_field], doc[self.target_field]
            outputs = self._inference(image)
            meter.update(output=outputs.cpu().numpy(), target=targets.cpu().numpy())
        miou_score, iou_dict = meter.m_iou()
        self.status['metrics'] = {'mIoU': miou_score}
        self.status['metrics'].update(iou_dict)
