#!/usr/bin/env python3

"""
@author: xi
@since: 2022-06-08
"""
import abc
import math
import random
from typing import Union, Sequence, Any, Callable, Type, Mapping, MutableMapping

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

__all__ = [
    'CosineDistance',
    'EuclideanDistance',
    'AbstractMemory',
    'BaseMemory',
    'PoolingMemory',
    'SoftKMeansMemory',
    'KExpansionMemory',
    'TailMemory',
    'HeadMemory',
    'HeadTailMemory',
    'MemoryBank'
]


class CosineDistance(nn.Module):
    """Cosine distance
    """

    def __init__(self, eps=1e-5):
        super(CosineDistance, self).__init__()
        self._eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        b = b.T
        sim = a @ b
        norm = torch.norm(a, 2, 1, keepdim=True) * torch.norm(b, 2, 0, keepdim=True)
        dist = -sim / (norm + self._eps)
        return dist


class EuclideanDistance(nn.Module):
    """Euclidean distance
    """

    def __init__(self, inplace=False):
        super(EuclideanDistance, self).__init__()
        self.inplace = inplace

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward
        """
        if self.inplace:
            bt = b.T
            dist = (a @ bt).mul_(-2.0)
            dist = dist.add_(a.square().sum(1, keepdim=True)).add_(bt.square().sum(0, keepdim=True))
            dist = dist.clip_(0.0).sqrt_()
        else:
            bt = b.T
            dist = a.square().sum(1, keepdim=True) + bt.square().sum(0, keepdim=True)
            dist = dist + (-2.0) * (a @ bt)
            dist = dist.clip(0.0).sqrt()
        # dist = torch.cdist(a[None, ...], b[None, ...], 2).squeeze(0)
        return dist


class AbstractMemory(abc.ABC):
    """Abstract memory
    """

    @abc.abstractmethod
    def read(self, *args, **kwargs):
        """Read
        """
        pass

    @abc.abstractmethod
    def write(self, *args, **kwargs):
        """Write
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset
        """
        pass

    @abc.abstractmethod
    def backup(self):
        """Backup
        """
        pass

    @abc.abstractmethod
    def restore(self):
        """Restore
        """
        pass


class BaseMemory(nn.Module, AbstractMemory):
    """Base memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            device=None,
            **kwargs
    ) -> None:
        """Abstract base class of updatable memory.

        Args:
            mem_size: the number of prototypes, i.e., "m".
            feat_size: the dimension of the prototype vectors, i.e., "d".
            device: the desired device of the prototypes.
        """
        super(BaseMemory, self).__init__()
        self.mem_size = mem_size
        self.feat_size = feat_size

        self.position = nn.Parameter(torch.empty((), dtype=torch.int64), requires_grad=False)
        self.prototypes = nn.Parameter(
            torch.empty(
                (self.mem_size, self.feat_size),
                dtype=torch.float32,
                device=device
            ),
            requires_grad=False
        )
        self.reset()

        self._backup = None

    def read(self) -> torch.Tensor:
        return self.prototypes.detach()

    def write(self, feat: torch.Tensor) -> Any:
        with torch.no_grad():
            position = int(self.position)
            if position == self.mem_size:
                return position  # no place to write

            num_feat = int(feat.shape[0])
            if num_feat == 0:
                return position  # nothing to write

            # append
            num_write = min(self.mem_size - position, num_feat)
            idx = np.linspace(0, feat.shape[0] - 1, num_write).astype(np.int64)
            np.random.shuffle(idx)
            idx = torch.from_numpy(idx)
            new_position = position + num_write
            self.prototypes[position:new_position, :] = feat[idx, :]
            self.position[...] = new_position

            position = new_position
            if position == self.mem_size:
                return position  # memory is already full

            # fill the memory
            num_repeat = self.mem_size // position
            num_rest = self.mem_size % position
            for i in range(1, num_repeat):
                start = i * position
                end = (i + 1) * position
                self.prototypes[start:end, :] = self.prototypes[:position, :]
            if num_rest > 0:
                self.prototypes[-num_rest:, :] = self.prototypes[:num_rest]
            return position

    def reset(self):
        nn.init.zeros_(self.position)
        nn.init.normal_(self.prototypes, 0.0, 1.0)

    def backup(self):
        with torch.no_grad():
            self._backup = (self.position.clone(), self.prototypes.clone())

    def restore(self):
        with torch.no_grad():
            if self._backup is not None:
                self.position[...] = self._backup[0]
                self.prototypes[...] = self._backup[1]

    def forward(self, feat: torch.Tensor = None):
        """Forward
        """
        return self.write(feat) if feat is not None else self.read()


class PoolingMemory(BaseMemory):
    """Pooling memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            device=None
    ) -> None:
        """A memory that updates its prototypes by pooling the input features.

        Args:
            mem_size: the number of prototypes, i.e., "m".
            feat_size: the dimension of the prototype vectors, i.e., "d".
            device: the desired device of the prototypes.
        """
        super(PoolingMemory, self).__init__(
            mem_size,
            feat_size,
            device=device
        )

    def write(self, feat: torch.Tensor) -> Any:
        if super(PoolingMemory, self).write(feat) < self.mem_size:
            return

        with torch.no_grad():
            if int(feat.shape[0]) == 0:
                return
            old = self.prototypes.T[None, :, :]
            new = F.adaptive_avg_pool1d(feat.T[None, :, :], self.mem_size)
            self.prototypes[:, :] = F.adaptive_avg_pool1d(torch.concat([old, new], 2), self.mem_size)[0].T


class SoftKMeansMemory(BaseMemory):
    """SOft kmeans memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            lr: float = 0.1,
            dist_fn: Union[str, Callable] = 'euclidean',
            sharpen: float = 0.5,
            eps: float = 1e-10,
            device=None
    ) -> None:
        """Soft K-means Memory.
        The prototype is represented as a (m, d) tensor.

        Args:
            mem_size: the number of prototypes, i.e., "m".
            feat_size: the dimension of the prototype vectors, i.e., "d".
            lr: the learning rate for every write operation.
            dist_fn: the distance function.
            device: the desired device of the prototypes.
        """
        super(SoftKMeansMemory, self).__init__(
            mem_size,
            feat_size,
            device=device
        )
        assert 0.0 < lr < 1.0
        self.lr = lr

        if isinstance(dist_fn, str):
            dist_fn = dist_fn.lower()
            if dist_fn == 'euclidean' or dist_fn == 'euc':
                self.dist_fn = EuclideanDistance()
            elif dist_fn == 'cosine' or dist_fn == 'cos':
                self.dist_fn = CosineDistance()
            else:
                raise RuntimeError(f'Unknown dist function "{dist_fn}".')
        else:
            assert callable(dist_fn)
            self.dist_fn = dist_fn

        self.sharpen = sharpen
        self.eps = eps
        self.device = device

        self.rank = None

    def write(self, feat: torch.Tensor) -> Any:
        if super(SoftKMeansMemory, self).write(feat) < self.mem_size:
            return

        with torch.no_grad():
            feat = torch.concat([feat, self.prototypes], 0)
            dist = self.dist_fn(self.prototypes, feat)  # (m, k)
            weight = F.softmax(-self.sharpen * dist, 0)
            update = weight @ feat / (weight.sum(1, keepdim=True) + self.eps)  # (m, d)
            lr = self.lr
            if self.rank is not None:
                rank = self.rank.float()[:, None]
                rmin, rmax = rank.min(), rank.max()
                lr = lr * (1.0 - (rank - rmin) / (rmax - rmin + self.eps))
            self.prototypes[:, :] = (1.0 - lr) * self.prototypes + lr * update

    def update_rank(self, rank: torch.Tensor) -> None:
        """Update rank
        """
        with torch.no_grad():
            if self.rank is None:
                self.rank = nn.Parameter(
                    torch.zeros((self.mem_size,), dtype=torch.float64, device=self.prototypes.device),
                    requires_grad=False
                )
            self.rank.add_(rank)


class KExpansionMemory(BaseMemory):
    """K-expansion memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            lr: float = 0.1,
            alpha: float = 0.1,
            k: int = 1,
            dist_fn: Union[str, Callable] = 'euclidean',
            device=None
    ) -> None:
        """K-expansion Memory.
        The algorithm is very similar to the "k-center problem".
        The prototype is represented as a (m, d) tensor.

        Args:
            mem_size: the number of prototypes, i.e., "m".
            feat_size: the dimension of the prototype vectors, i.e., "d".
            lr: the learning rate for every write operation.
            alpha: the proportion of the prototypes to update for every class.
            dist_fn: the distance function.
            device: the desired device of the prototypes.
        """
        super(KExpansionMemory, self).__init__(
            mem_size,
            feat_size,
            device=device
        )
        assert 0.0 <= lr <= 1.0
        self.lr = lr

        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

        assert 1 <= k <= self.mem_size
        self.k = k

        if isinstance(dist_fn, str):
            dist_fn = dist_fn.lower()
            if dist_fn == 'euclidean' or dist_fn == 'euc':
                self.dist_fn = EuclideanDistance()
            elif dist_fn == 'cosine' or dist_fn == 'cos':
                self.dist_fn = CosineDistance()
            else:
                raise RuntimeError(f'Unknown dist function "{dist_fn}".')
        else:
            assert callable(dist_fn)
            self.dist_fn = dist_fn

    def write(self, feat: torch.Tensor) -> Any:
        if super(KExpansionMemory, self).write(feat) < self.mem_size:
            return

        with torch.no_grad():
            dist = self.dist_fn(feat, self.prototypes)  # (k, m)
            if self.k == 1:
                dist_to_memory, nearest_proto_idx = torch.min(dist, 1)  # (k,)
            else:
                dist_to_memory, nearest_proto_idx = torch.topk(dist, self.k, 1, largest=False)  # (k, topk)
                memory_density = self.dist_fn(self.prototypes, self.prototypes).sum(1).neg()
                nearest_proto_density = memory_density[nearest_proto_idx]
                _, index = torch.max(nearest_proto_density, 1, keepdim=True)  # (k, 1)
                # beta_max = self.k * 5
                # beta_min = 3
                # beta = (1 - self.lr) * (beta_max - beta_min) + beta_min
                # index = Beta(1, beta).sample((feat.shape[0], 1)).to(feat.device)  # (k, 1)
                # index = (index * self.k).long()  # (k, 1)
                # dist_to_memory = dist_to_memory.gather(1, index)
                dist_to_memory = dist_to_memory[:, 0]
                nearest_proto_idx = nearest_proto_idx.gather(1, index).squeeze(1)

            # "alpha" indicates how many prototypes in the memory will be updated.
            # Note that the update is based on (query, proto) pairs given by "dist_to_memory", so the update number
            # cannot exceed the number of queries.
            num_update = min(math.ceil(self.mem_size * self.alpha), dist_to_memory.shape[0])
            farthest_query_idx = torch.topk(dist_to_memory, num_update)[1]  # (num_update,)

            update_proto_idx = nearest_proto_idx[farthest_query_idx]  # (num_update,)
            update = self.prototypes[update_proto_idx, :]  # (num_update, d)
            update.mul_(1.0 - self.lr).add_(feat[farthest_query_idx, :], alpha=self.lr)
            self.prototypes[update_proto_idx, :] = update

            mean = dist_to_memory.mean()
            var = dist_to_memory.square().mean() - mean.square()
            return mean, var


class TailMemory(BaseMemory):
    """Tail memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            lr: float = 0.1,
            alpha: float = 0.1,
            dist_fn: Union[str, Callable] = 'euclidean',
            w_var: float = 0.0,
            device=None
    ) -> None:
        super(TailMemory, self).__init__(
            mem_size,
            feat_size,
            device=device
        )
        assert 0.0 <= lr <= 1.0
        self.lr = lr

        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

        assert 0.0 <= w_var <= 1.0
        self.w_var = w_var

        if isinstance(dist_fn, str):
            dist_fn = dist_fn.lower()
            if dist_fn == 'euclidean' or dist_fn == 'euc':
                self.dist_fn = EuclideanDistance()
            elif dist_fn == 'cosine' or dist_fn == 'cos':
                self.dist_fn = CosineDistance()
            else:
                raise RuntimeError(f'Unknown dist function "{dist_fn}".')
        else:
            assert callable(dist_fn)
            self.dist_fn = dist_fn

    def write(self, feat: torch.Tensor) -> Any:
        if super(TailMemory, self).write(feat) < self.mem_size:
            return

        with torch.no_grad():
            dist = self.dist_fn(feat, self.prototypes)  # (k, m)
            dist_to_memory, nearest_proto_idx = torch.min(dist, 1)  # (k,)

            num_update = min(math.ceil(self.mem_size * self.alpha), dist_to_memory.shape[0])
            farthest_query_idx = torch.topk(dist_to_memory, num_update)[1]  # (num_update,)

            if self.lr > 0.5:
                replace_query_idx = farthest_query_idx[0]
                farthest_query_idx = farthest_query_idx[1:]
                replace_proto_idx = random.randint(0, self.mem_size - 1)
                self.prototypes[replace_proto_idx, :] = feat[replace_query_idx, :]

            update_proto_idx = nearest_proto_idx[farthest_query_idx]  # (num_update,)
            update = self.prototypes[update_proto_idx, :]  # (num_update, d)
            if self.w_var > 0.0:
                mean = dist_to_memory.mean()
                a = self.lr * (1 + self.w_var * (dist_to_memory[farthest_query_idx] - mean))[:, None]  # (num_update, d)
                update.mul_(1.0 - a).add_(feat[farthest_query_idx, :] * a)
            else:
                update.mul_(1.0 - self.lr).add_(feat[farthest_query_idx, :], alpha=self.lr)
            self.prototypes[update_proto_idx, :] = update

            # var = dist_to_memory.square().mean() - mean.square()
            # return mean, var


class HeadMemory(BaseMemory):
    """Head memory
    """

    def __init__(
            self,
            mem_size: int,
            feat_size: int,
            alpha=0.1,
            device=None
    ) -> None:
        super(HeadMemory, self).__init__(
            mem_size,
            feat_size,
            device=device
        )

        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

    def write(self, feat: torch.Tensor) -> Any:
        if super(HeadMemory, self).write(feat) < self.mem_size:
            return

        with torch.no_grad():
            num_update = min(math.ceil(self.mem_size * self.alpha), feat.shape[0])
            farthest_query_idx = torch.randperm(feat.shape[0])[:num_update]

            update_proto_idx = torch.randperm(self.mem_size)[:num_update]
            self.prototypes[update_proto_idx, :] = feat[farthest_query_idx, :]


class HeadTailMemory(nn.Module, AbstractMemory):
    """Head tail memory
    """

    def __init__(
            self,
            mem_size,
            feat_size,
            lr=0.1,
            alpha=0.2,
            dist_fn: Union[str, Callable] = 'euclidean',
            w_var=0.0,
            device=None
    ) -> None:
        super(HeadTailMemory, self).__init__()
        mem_size1 = int(mem_size * 0.5)
        mem_size2 = mem_size - mem_size1
        self.head = HeadMemory(mem_size1, feat_size, alpha=alpha, device=device)
        self.tail = TailMemory(
            mem_size2,
            feat_size,
            lr=lr,
            alpha=alpha,
            dist_fn=dist_fn,
            w_var=w_var,
            device=device
        )

    @property
    def lr(self):
        """Learning rate
        """
        return self.tail.lr

    @lr.setter
    def lr(self, lr):
        self.tail.lr = lr

    @property
    def prototypes(self):
        """Prototypes
        """
        return torch.concat([self.head.prototypes, self.tail.prototypes])

    def read(self):
        return torch.concat([self.head.read(), self.tail.read()])

    def write(self, feat):
        self.head.write(feat)
        return self.tail.write(feat)

    def reset(self):
        self.head.reset()
        self.tail.reset()

    def backup(self):
        self.head.backup()
        self.tail.backup()

    def restore(self):
        self.head.restore()
        self.tail.restore()

    def forward(self, feat: torch.Tensor = None):
        """Forward
        """
        return self.write(feat) if feat is not None else self.read()


class MemoryBank(nn.Module, AbstractMemory):
    """Memory bank
    """

    def __init__(
            self,
            MemoryType: Type,
            mem_size: int,
            feat_size: int,
            num_classes: int,
            memory_kwargs: Mapping[str, Any]
    ) -> None:
        """A memory bank that contains memories for multiple tasks.

        Args:
            MemoryType: the memory class or construct function.
            mem_size: the number of prototypes, i.e., "m".
            feat_size: the dimension of the prototype vectors, i.e., "d".
            num_classes: the number of classes, i.e., "c".
            memory_kwargs: the extra arguments to construct the memory, such as learning rate, dist functions.
        """
        super(MemoryBank, self).__init__()
        self.MemoryType = MemoryType
        self.mem_size = mem_size
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.memory_kwargs = memory_kwargs

        self.memories = nn.ModuleDict()  # type: MutableMapping[str, nn.ModuleList]

    def read(self, task_name: str) -> torch.Tensor:
        assert task_name in self.memories
        return torch.stack([
            memory.read()
            for memory in self.memories[task_name]
        ])

    def write(self, task_name: str, feat_list: Sequence[torch.Tensor]) -> None:
        if task_name not in self.memories:
            memories = nn.ModuleList()
            self.memories[task_name] = memories
            for i in range(self.num_classes):
                memories.append(self.MemoryType(
                    mem_size=self.mem_size,
                    feat_size=self.feat_size,
                    device=feat_list[i].device,
                    **self.memory_kwargs
                ))
        for memory, feat in zip(self.memories[task_name], feat_list):
            memory.write(feat)

    def reset(self):
        self.memories.clear()

    def backup(self):
        for memories in self.memories.values():
            for memory in memories:
                memory.backup()

    def restore(self):
        for memories in self.memories.values():
            for memory in memories:
                memory.restore()

    def forward(self, task_name: str, feat: torch.Tensor = None):
        """Forward
        """
        return self.write(task_name, feat) if feat is not None else self.read(task_name)

    def call(self, fn_name, task_name, *args):
        """Call
        """
        assert task_name in self.memories
        results = []
        for i in range(self.num_classes):
            inner_i = getattr(self.memories[task_name][i], fn_name)
            args_i = [arg[i] for arg in args]
            results.append(inner_i(*args_i))
        return results


class MemoryBankClassification(MemoryBank):
    """Memory bank classification
    """
    def __init__(self,
                 MemoryType: Type[BaseMemory],
                 mem_size: int,
                 feat_size: int,
                 num_classes: int,
                 device: str,
                 task_name: str,
                 base_lr: float,
                 memory_kwargs: Mapping[str, Any],
                 ):
        super(MemoryBankClassification, self).__init__(
            MemoryType,
            mem_size,
            feat_size,
            num_classes,
            memory_kwargs
        )
        self.task_name = task_name
        self.device = device
        self.memories[task_name] = nn.ModuleList()
        self._init_memory()
        self.dist_fn = CosineDistance()
        self.base_lr = base_lr

    def _init_memory(self):
        for _ in range(self.num_classes):
            self.memories[self.task_name].append(self.MemoryType(
                mem_size=self.mem_size,
                feat_size=self.feat_size,
                device=self.device,
                **self.memory_kwargs
            ))

    def write_episode(self, task_episode, supp_feat_episode):
        """
        task_episode : (num_shot)
        supp_feat_episode : (p, c, h, w)
        """
        for i, task in enumerate(task_episode):
            # supp_feat = supp_feat_episode[i]  # (c, h, w)
            supp_feat = supp_feat_episode[i].reshape(self.feat_size, -1).T  # (k, d)
            memory = self.memories[self.task_name][task]
            memory.write(supp_feat)

    def update(self, query_feat, task):
        """
       supp_feat: (n, c, h, w)
       task: tensor (n)
       """
        query_feat = rearrange(query_feat, 'n c h w -> n c (h w)')  # (n c k)
        query_feat = query_feat.permute(0, 2, 1)  # (n, k, d)
        for i, task_i in enumerate(task):
            memory = self.memories[self.task_name][task_i]
            memory.write(query_feat[i])

    def forward(self, supp_feat, task):
        """
        supp_feat: (np, c, h, w)
        task: tensor (n, num_shot)
        return (c, m, d)
        """
        supp_feat = rearrange(supp_feat, '(n p) c h w -> n p c h w', n=task.shape[0])
        for batch_index in range(task.shape[0]):
            task_episode = task[batch_index]  # (num_shot)
            supp_feat_episode = supp_feat[batch_index]  # (num_way*num_shot, c, h, w)
            self.write_episode(task_episode, supp_feat_episode)
        return self.read(self.task_name)


class MemoryBankMultiLabel(MemoryBank):
    """Memory bank for multi-label
    """
    def __init__(self,
                 MemoryType: Type[BaseMemory],
                 mem_size: int,
                 feat_size: int,
                 num_classes: int,
                 device: str,
                 task_name: str,
                 base_lr: float,
                 memory_kwargs: Mapping[str, Any],
                 ):
        super(MemoryBankMultiLabel, self).__init__(
            MemoryType,
            mem_size,
            feat_size,
            num_classes,
            memory_kwargs
        )
        self.task_name = task_name
        self.device = device
        self.memories[task_name] = nn.ModuleList()
        self._init_memory()
        self.dist_fn = CosineDistance()
        self.base_lr = base_lr

    def _init_memory(self):
        for _ in range(self.num_classes):
            self.memories[self.task_name].append(self.MemoryType(
                mem_size=self.mem_size,
                feat_size=self.feat_size,
                device=self.device,
                **self.memory_kwargs
            ))

    def write_episode(self, task_onehot, supp_feat_i):
        """
        task_onehot : (num_class)
        supp_feat_i : (c, h, w)
        """
        memory_update_list = []
        memory_name_list = []
        supp_feat_i = supp_feat_i.reshape((supp_feat_i.shape[0], -1)).T  # (k, d)
        for task_name in range(self.num_classes):
            if task_onehot[task_name] == 1:
                memory_update_list.append(self.memories[self.task_name][task_name].prototypes)
                memory_name_list.append(task_name)
        mem_update = torch.stack(memory_update_list)  # (n, m, d)
        dist = self.dist_fn.high_dim_dist(mem_update, supp_feat_i)  # (n, m, k)
        dist, _ = torch.min(dist, dim=1)
        _, min_index = torch.min(dist, dim=0)  # (k)
        lr = self.base_lr / len(memory_name_list)
        for k, index in enumerate(min_index):
            feat = supp_feat_i[k].unsqueeze(0)  # (1, d)
            memory = self.memories[self.task_name][memory_name_list[index]]
            memory.lr = lr
            memory.write(feat)

    def forward(self, supp_feat, task):
        """
        supp_feat: (np, c, h, w)
        task: tensor (n, num_shot)
        return (c, m, d)
        """
        num_shot = task.shape[1]
        supp_feat = rearrange(supp_feat, '(n p) c h w -> n p c h w', n=task.shape[0])
        for batch_index in range(task.shape[0]):
            task_episode = task[batch_index]  # (num_way*num_shot, num_class)
            supp_feat_episode = supp_feat[batch_index]  # (num_way*num_shot, c, h, w)
            for i in range(num_shot):
                task_onehot = task_episode[i]  # (num_class)
                supp_feat_i = supp_feat_episode[i]  # (c, h, w)
                self.write_episode(task_onehot, supp_feat_i)
        return self.read(self.task_name)
