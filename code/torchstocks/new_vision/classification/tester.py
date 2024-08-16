#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.new_common.tester import AbstractTester, init_test_dataset
from torchstocks.utils.metrics import ClassificationMeter

__all__ = [
    'ClassificationTester'
]


class ClassificationTester(AbstractTester):
    """Classification tester
    """

    def __init__(
            self,
            model: nn.Module,
            test_dataset: Dataset,
            batch_size: int = 256,
            num_workers: int = 10,
            input_field: str = 'image',
            target_field: str = 'label',
            device: str = 'cpu',
            output_dir: str = None
    ) -> None:
        super(ClassificationTester, self).__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_field = input_field
        self.target_field = target_field
        self.device = device
        self.output_dir = output_dir

        self.test_dataset, self.test_loader, self.test_transform = init_test_dataset(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def _inference(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.to(self.device)
            y_ = self.model(x).argmax(-1)
            return y_.detach().cpu()

    def __call__(self, doc):
        if callable(self.test_transform):
            doc = self.test_transform(doc)
        x = doc[self.input_field]
        x = torch.as_tensor(x).unsqueeze(0)
        y_ = self._inference(x)
        y_ = y_.squeeze(0).numpy()
        return {self.target_field: y_}

    def run(self):
        if self.test_loader is None:
            return

        meter = ClassificationMeter()
        self.model.eval()
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            x, y = doc[self.input_field], doc[self.target_field]
            y_ = self._inference(x)
            meter.update(output=y_.numpy(), target=y.numpy())
        precision, recall, f1 = meter.precision_recall_f1()
        indx = list(range(0, len(precision)))
        precision_dict = dict(zip(indx, precision))
        recall_dict = dict(zip(indx, recall))
        f1_dict = dict(zip(indx, f1))

        self.status['metrics'] = {
            'Acc': meter.accuracy(),
            'F1': f1.mean(),
            'F1_every_class': f1_dict,
            'Precision': precision.mean(),
            'Precision_every_class': precision_dict,
            'Recall': recall.mean(),
            'Recall_every_class': recall_dict
        }
