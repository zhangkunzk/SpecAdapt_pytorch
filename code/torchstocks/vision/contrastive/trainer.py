#!/usr/bin/env python3


from torch import nn
from torch.utils.data import Dataset

from torchstocks.common.contrastive.trainer import ContrastiveTrainer

__all__ = [
    'ImageContrastiveTrainer'
]


class ImageContrastiveTrainer(ContrastiveTrainer):

    def __init__(
            self,
            model: nn.Module,
            unlabeled_dataset: Dataset,
            train_dataset: Dataset,
            test_dataset: Dataset,
            image_field: str = 'image',
            label_field: str = 'label',
            optimizer: str = 'AdamW',
            batch_size: int = 256,
            max_lr: float = 1e-3,
            momentum: float = 0.9,
            weight_decay: float = 0.3,
            num_epochs: int = 100,
            num_workers: int = 10,
            param_groups: list = None,
            clip_grad_norm: float = 0.1,
            eval_interval: int = 5,
            device: str = 'cpu',
            eval_method: str = 'logistic'
    ) -> None:
        super(ImageContrastiveTrainer, self).__init__(
            model=model,
            unlabeled_dataset=unlabeled_dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            input_field=image_field,
            label_field=label_field,
            optimizer=optimizer,
            batch_size=batch_size,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_workers=num_workers,
            param_groups=param_groups,
            clip_grad_norm=clip_grad_norm,
            eval_interval=eval_interval,
            device=device,
            eval_method=eval_method
        )
