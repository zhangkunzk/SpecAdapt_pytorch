# oml-pytorch

Pytorch implementation of OML( [paper](https://proceedings.neurips.cc/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf), [code](https://github.com/khurramjaved96/mrcl))

Reference:  ANML([paper](https://arxiv.org/abs/2002.09571), [code1](https://github.com/uvm-neurobotics-lab/higherANML), [code2](https://github.com/uvm-neurobotics-lab/ANML))

## Dateset

```python
omniglot/
    train/
    test/
```

## Train

|phase|  trained module| optimazer|   lr  |
| :-: | :-: | :-: | :-: |
|inner loop        |   head   |   sgd   |   0.1   |
|    outer loop    |   backbone+head    |  adamw   |  0.001    |

network: (6 Conv layers + 1 linear layer) (backbone) + 1 linear layer (head).
## Test
|    phase    |  trained module    |  optimazer    |   lr   |
| :-: | :-: | :-: | :-: |
|inner loop        |   head   |   adam   |   0.0005   |

In meta test,  the result is averaged  over multiple runs.  Default: 10 times.

15 samples for meta-test-train (only pass once sequentially ) and 5 samples for evaluate accuracy in every task.

|number of tasks|accuracy|
| :-: | :-: |
|200|67.62%|
|150|71.48%|
|100|77.76%|
|50|86.19%|
|10|96.40%|


## Run script

```shell
CUDA_VISIBLE_DEVICES='3' python -m torchstocks.vision.meta_learning.online_class_incremental.train --data_path '/mnt/cephfs/data/omniglot/class_docset/' --model torchstocks.vision.meta_learning.online_class_incremental.model.oml.Model --network torchstocks.vision.meta_learning.online_class_incremental.model.oml.ConvNetwork
```