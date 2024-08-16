#!/usr/bin/env python3


from entry import TrainOptions, TrainEntry, DATASET_DICT
from torchstocks.utils import ArgumentParser, Experiment

PARAM_GROUPS = [
    {"name": "model.backbone", "lr": 0},
    {
        "match": [
            # "^.*patch_embed.proj.weight$",
            # "^.*(blocks).*(attn.(qkv|proj).*weight)$",
            # "^.*mlp.*weight$",
            "^.*weight$",
            "^.*bias$"
        ],
        "tag": "low_rank"
    },
    {"name": "model.backbone.head"}
]


def main():
    options = TrainOptions()
    options.model = 'model.vit.Model'
    options.batch_size = 64
    options.num_epochs = 100
    options.optimizer = 'SAdamW'
    options.max_lr = 3e-3
    options.weight_decay = 0.3
    options.param_groups = PARAM_GROUPS
    options.lr_decay_min_value = 0.0
    options.drop_path_rate = 0.1

    parser = ArgumentParser()
    parser.add_argument('--dataset_name')
    parser.add_argument('--rank', '-r', type=int, required=True)
    parser.add_argument('--exp_dir')
    args = parser.parse_args()
    options.update(args)

    exp = Experiment(args.exp_dir) if args.exp_dir is not None else None

    if args.dataset_name is not None:
        entry = TrainEntry(options)
        entry.train()

        if exp is not None:
            metric = entry.trainer.get_status('metrics')
            exp.log('result.txt', f'{args.dataset_name} {metric}')
    else:
        for dataset_name in DATASET_DICT:
            options.dataset_name = dataset_name
            entry = TrainEntry(options)
            entry.train()

            if exp is not None:
                metric = entry.trainer.get_status('metrics')
                exp.log('result.txt', f'{dataset_name} {metric}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
