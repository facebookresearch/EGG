# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
from typing import Iterable, List, Tuple

import numpy as np
import torch


class AttributesValuesDataset:
    """
    >>> d1 = AttributesValuesDataset(3, 4, 20, 5)
    >>> epoch1 = [batch for batch in d1]
    >>> len(epoch1)
    4
    >>> epoch1[0][0].shape
    torch.Size([5, 3])
    """
    def __init__(self,
                 n_attributes: int,
                 n_values: int,
                 n_train_samples_per_epoch: int,
                 batch_size: int,
                 seed: int = 111):

        self.n_attributes = n_attributes
        self.n_values = n_values

        self.n_train_samples_per_epoch = n_train_samples_per_epoch
        self.batch_size = batch_size

        self.n_batches_per_epoch = int(n_train_samples_per_epoch) // batch_size
        assert self.n_batches_per_epoch

        self.samples = list(itertools.product(*(range(1, n_values+1) for _ in range(n_attributes))))

        self.seed = seed

    def __iter__(self):
        return AttributesValuesIterator(self.batch_size, self.n_batches_per_epoch, self.samples, self.seed)


class AttributesValuesIterator:
    """
    >>> samples = list(itertools.product(*(range(1, 4) for _ in range(3))))
    >>> it1 = AttributesValuesIterator(10, 2, samples)
    >>> it2 = AttributesValuesIterator(10, 2, samples)
    >>> l1 = [batch[0] for batch in it1]
    >>> l2 = [batch[0] for batch in it1]
    >>> all([v1.allclose(v2) for v1, v2 in zip(l1, l2)])
    True
    """
    def __init__(self,
                 batch_size: int,
                 n_batches_per_epoch: int,
                 samples: Iterable,
                 seed: int = 111):

        self.batch_size = batch_size
        self.n_batches_per_epoch = n_batches_per_epoch

        self.batches_generated = -1
        self.idx = 0

        self.data = samples
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.batches_generated += 1

        if self.idx + self.batch_size >= len(self.data) or self.batches_generated >= self.n_batches_per_epoch:
            self.batches_generated = 0
            self.idx = 0
            raise StopIteration()

        if not self.batches_generated:
            self.data = self.random_state.permutation(self.data)

        tnsr = [torch.Tensor(elem) for elem in self.data[self.idx:self.idx+self.batch_size]]
        self.idx += self.batch_size
        batch = torch.stack(tnsr)
        labels = torch.zeros(1)

        return batch, labels
