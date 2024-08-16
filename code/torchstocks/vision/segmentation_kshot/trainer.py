#!/usr/bin/env python3

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from torchstocks.common.trainer import BPTrainerWithDataset
from torchstocks.utils.metrics import IouMeter
from torchstocks.vision.segmentation_kshot.dataset import IGNORE_CLASS

__all__ = [
    'SegmentationBinaryKShotTrainer'
]


class SegmentationBinaryKShotTrainer(BPTrainerWithDataset):

    def __init__(
            self,
            image_size: int,
            model: nn.Module,
            train_dataset: Dataset,
            test_dataset: Dataset,
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            param_groups: list = None,  # todo
            clip_grad_norm: float = 0.1,
            device: str = 'cpu',
            input_field: str = 'image',
            mask_field: str = 'mask',
            label_field: str = 'label',
            output_dir: str = None
    ) -> None:
        super(SegmentationBinaryKShotTrainer, self).__init__(
            model=model,
            train_dataset=train_dataset,
            auxiliary_dataset=test_dataset,
            optimizer=optimizer,
            batch_size=batch_size,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            device=device
        )
        self.test_loader = self.auxiliary_loader
        self.image_size = image_size
        self.input_field = input_field
        self.mask_field = mask_field
        self.label_field = label_field
        self.output_dir = output_dir

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

    def train_step(self, task, sx, sy, qx, qy):
        sx = sx.to(self.device)
        sy = sy.to(self.device)
        qx = qx.to(self.device)
        qy = qy.to(self.device)
        sy[torch.where(torch.eq(sy, IGNORE_CLASS))] = 0  # clear the "ignore" class
        qy[torch.where(torch.eq(qy, IGNORE_CLASS))] = 0  # clear the "ignore" class
        loss = self.model((task, sx, sy, qx), qy)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.detach().cpu()

    def predict_step(self, task, sx, sy, qx):
        with torch.no_grad():
            sx = sx.to(self.device) if sx is not None else None
            sy = sy.to(self.device) if sy is not None else None
            qx = qx.to(self.device)
            if sy is not None:
                sy[torch.where(torch.eq(sy, IGNORE_CLASS))] = 0  # clear the "ignore" class
            output = self.model((task, sx, sy, qx))  # (n, num_classes, ?, ?)
            output = F.interpolate(output, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
            qy_ = torch.argmax(output, 1)
            return qy_.detach().cpu()

    def train(self):
        if self.train_loader is None:
            return

        self.status['loop'] = 0
        loss_g = None
        for epoch in range(self.num_epochs):
            self.status['epoch'] = epoch + 1
            self.model.train()
            self.model.reset_memories()  # todo
            loop = tqdm(self.train_loader, leave=False, ncols=96)
            for query_doc, supp_doc in loop:
                loss = self.train_step(
                    task=[str(int(c)) for c in query_doc[self.label_field]],
                    sx=supp_doc[self.input_field],
                    sy=supp_doc[self.mask_field],
                    qx=query_doc[self.input_field],
                    qy=query_doc[self.mask_field]
                )
                if loss_g is None:
                    loss_g = loss
                loss_g = 0.9 * loss_g + 0.1 * loss
                lr = self.optimizer.param_groups[0]['lr']

                self.status['loop'] += 1
                self.status['loss'] = loss
                self.status['loss_g'] = loss_g
                self.status['lr'] = lr

                info = f'[{epoch + 1}/{self.num_epochs}] L={loss_g:.06f} LR={lr:.02e}'
                loop.set_description(info, False)

            self.model.eval()
            self.model.reset_memories()
            self.evaluate(use_supp=True)
            m_iou, fb_iou = self.status['metrics']['m_iou'], self.status['metrics']['fb_iou']
            # self.evaluate(use_supp=False)
            # m_iou1, fb_iou1 = self.status['metrics']['m_iou'], self.status['metrics']['fb_iou']
            print(
                f'[{epoch + 1}/{self.num_epochs}] '
                f'L={loss_g:.06f} '
                f'mIOU={m_iou:.02%} '
                f'fbIOU={fb_iou:.02%} '
            )

            if self.output_dir is not None:
                path = os.path.join(self.output_dir, 'model.pth')
                torch.save(self.model, path)

    def evaluate(self, use_supp=True):
        if self.test_loader is None:
            return

        meter = IouMeter(IGNORE_CLASS)
        loop = tqdm(self.test_loader, ncols=96, leave=False)
        for i, (query_doc, supp_doc) in enumerate(loop):
            image = query_doc[self.input_field].numpy()
            output = self.predict_step(
                task=[str(int(c)) for c in query_doc[self.label_field]],
                sx=supp_doc[self.input_field] if use_supp else None,
                sy=supp_doc[self.mask_field] if use_supp else None,
                qx=query_doc[self.input_field],
            ).numpy()
            target = query_doc[self.mask_field].numpy()  # (n, h, w)
            class_list = [int(c) for c in query_doc[self.label_field]]
            size_list = [(int(size), int(size)) for size in query_doc['size']] if 'size' in query_doc else None

            meter.update(output, target, class_list, size_list)
            loop.set_description(f'mIOU={meter.m_iou():.02%}')

            # if self.output_dir is not None:
            #     for j, (image_i, label_i) in enumerate(zip(image, output)):
            #         image_i = denormalize_image(image_i, transpose=True)
            #         label_i = dataset.decode_label(label_i)
            #         image_with_mask = draw_mask(image_i, label_i)
            #         image_with_mask = np.flip(image_with_mask, 2)  # RGB to BGR
            #         output_path = os.path.join(self.output_dir, f'{i:04d}-{j:04d}.jpg')
            #         cv.imwrite(output_path, image_with_mask)

        self.status['metrics'] = {
            'm_iou': meter.m_iou(),
            'fb_iou': meter.fb_iou()
        }
