#!/usr/bin/env python3


import os
import cv2 as cv
import torch

from torchstocks.common.entry import AbstractModelEntry
from torchstocks.utils import fix_random_seed, ArgumentParser, Config, literal_eval, import_by_name, \
    save_model, load_model
from util import get_classes_num
from dataset import get_dataset
from trainer import ClassificationTrainer


cv.setNumThreads(0)
fix_random_seed(42)


class ModelEntry(AbstractModelEntry):

    def __init__(self, args):
        super().__init__(args)

        load = literal_eval(args.load)
        if isinstance(load, str):
            print(f'Load model from {args.load}.')
            model = torch.load(args.load)
        else:
            model_config = Config(import_by_name(args.model), args)
            model_config.num_classes = get_classes_num(args.dataset_name)
            print(f'==== Model config ====\n{repr(model_config)}\n')
            model = model_config.build()
            load_model(model, load)
        if args.load_state_dict:
            loaded_model = torch.load(args.load_state_dict, map_location='cpu')
            if args.load_submodule:
                # 'this submodule': 'load submodule'
                # {'model.backbone': 'network.backbone'}
                assert isinstance(args.load_submodule, dict)
                for key, value in args.load_submodule.items():
                    origin_module = model.get_submodule(key)
                    updated_module = loaded_model.get_submodule(value)
                    origin_module.load_state_dict(updated_module.state_dict())
            else:
                model_state_dict = loaded_model.state_dict()
                model.load_state_dict(model_state_dict)
            print(f'<<<<<Load state_dict from {args.load_state_dict}>>>>>')
        self.model = model

    def train(self):
        args = self.args

        ################################################################################
        # create dataset
        ################################################################################
        train_dataset = None
        test_dataset = None
        if args.dataset_name:
            train_dataset, test_dataset = get_dataset(args.dataset_name, args.image_size)

        ################################################################################
        # create trainer
        ################################################################################
        trainer_config = Config(ClassificationTrainer, args)
        trainer_config.model = self.model
        trainer_config.train_dataset = train_dataset
        trainer_config.test_dataset = test_dataset
        trainer_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'==== Trainer config ====\n{trainer_config}\n')
        trainer = trainer_config.build()
        if train_dataset:
            trainer.train()
        elif test_dataset:
            trainer.evaluate()
        else:
            print('Nothing to do.')

        ################################################################################
        # save model
        ################################################################################
        save_model(self.model, literal_eval(args.save))
        os.makedirs('output', exist_ok=True)
        with open(f'output/{args.log_name}', 'a') as f:
            metric = trainer.get_status('metrics')
            f.write(f'{args.dataset_name} {metric} \n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)

    parser.add_argument('--model', required=True)
    parser.add_argument('--load', default=None)
    parser.add_argument('--save', default=None)

    parser.add_argument('--load_state_dict', default=None, help='only load state_dict')
    parser.add_argument('--load_submodule', type=eval, default=None, help='orginal model <- loaded model')

    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--param_groups', type=eval, default=None)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--log_name', type=str, default='result.txt')

    parser.add_argument('--note')
    args = parser.parse_args()

    ModelEntry(args).train()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
