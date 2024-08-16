#!/usr/bin/env python3

from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.new_common.tester import AbstractTester, init_test_dataset
from torchstocks.utils.metrics import MAPMeter
from torchstocks.utils.image import normalize_image
from torchstocks.common.dataset import DataCollate


__all__ = [
    'DetectionTester'
]

class DetectionTester(AbstractTester):
    """Detection tester
    """
    def __init__(
            self,
            model: nn.Module,
            decoder: nn.Module,
            test_dataset: Dataset,
            test_collate: DataCollate,
            batch_size: int = 256,
            num_workers: int = 10,
            input_field: str = 'image',
            target_field: str = 'bboxes',
            device: str = 'cpu',
            output_dir: str = None
    ) -> None:
        super(DetectionTester, self).__init__()
        self.model = model
        self.model.eval()
        self.decoder = decoder
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

    def _inference(self, image: torch.Tensor) -> List:
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(inputs=image, targets=None)
            if hasattr(self.model, 'num_heads'):
                outputs = self.decoder(outputs)
            elif hasattr(self.model, 'in_features'):
                boxes, scores, image_shapes = outputs
                outputs = self.decoder(boxes, scores, image_shapes)
                outputs = [torch.cat((res['pred_boxes'], res['pred_classes'][:, None],
                                    res['scores'][:, None]), dim=1) for res in outputs]
            else:
                print('error')
            outputs = [output.detach().cpu() for output in outputs]
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
        if hasattr(self.model, 'num_heads'):
            data_format = 'xywh'
        elif hasattr(self.model, 'in_features'):
            data_format='xyxy'
        else:
            print('error')
        meter = MAPMeter(0.5, data_format=data_format)
        loop = tqdm(self.test_loader, leave=False, ncols=96)
        for doc in loop:
            filename = doc['filename']
            image, bboxes = doc[self.input_field], doc[self.target_field]
            outputs = self._inference(image)
            bboxes = [bbox.numpy() for bbox in bboxes]
            outputs = [output.numpy() for output in outputs]
            for idx, file_name in enumerate(filename):
                meter.update(file_name, output=outputs[idx], target=bboxes[idx])
        map_score, ap_dict = meter.m_ap()
        f1_curve_dict = meter.m_f1_curve()
        ap_dict.update({'mAP': map_score})
        self.status['metrics'] = {'AP50': ap_dict}
        self.status['metrics'].update({'f1_score': f1_curve_dict})
        # self.status['metrics'] 数据格式：
            # {
            #     'AP50': {0: float, 1: float, ..., 'mAP': float},
            #     'f1_score': {0: np.ndarray, 1: np.ndarray, ..., 'mean': np.ndarray}
            # }
