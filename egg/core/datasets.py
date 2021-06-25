# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Iterable, Tuple

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

    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        n_train_samples_per_epoch: int,
        batch_size: int,
        seed: int = None,
    ):

        self.n_attributes = n_attributes
        self.n_values = n_values

        self.n_train_samples_per_epoch = n_train_samples_per_epoch
        self.batch_size = batch_size

        self.n_batches_per_epoch = int(n_train_samples_per_epoch) // batch_size
        assert self.n_batches_per_epoch

        self.samples = list(
            itertools.product(*(range(1, n_values + 1) for _ in range(n_attributes)))
        )

        seed = seed if seed else np.random.randint(0, 2 ** 31)
        self.seed = seed

    def __iter__(self):
        return AttributesValuesIterator(
            self.samples, self.batch_size, self.n_batches_per_epoch, self.seed
        )


class AttributesValuesIterator:
    """
    >>> samples = list(itertools.product(*(range(1, 4) for _ in range(3))))
    >>> it1 = AttributesValuesIterator(samples, 10, 2, 11)
    >>> it2 = AttributesValuesIterator(samples, 10, 2, 11)
    >>> l1 = [batch[0] for batch in it1]
    >>> l2 = [batch[0] for batch in it2]
    >>> all([v1.allclose(v2) for v1, v2 in zip(l1, l2)])
    True
    """

    def __init__(
        self,
        samples: Iterable,
        batch_size: int,
        n_batches_per_epoch: int,
        seed: int = None,
    ):

        self.batch_size = batch_size
        self.n_batches_per_epoch = n_batches_per_epoch

        self.batches_generated = 0
        self.idx = 0

        self.data = samples
        seed = seed if seed else np.random.randint(0, 2 ** 31)
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.idx + self.batch_size >= len(self.data)
            or self.batches_generated >= self.n_batches_per_epoch
        ):
            self.batches_generated = 0
            self.idx = 0
            raise StopIteration()

        if not self.batches_generated:
            self.data = torch.from_numpy(
                self.random_state.permutation(self.data)
            ).float()

        tnsr = [
            torch.Tensor(elem)
            for elem in self.data[self.idx : self.idx + self.batch_size]
        ]
        self.idx += self.batch_size - 1
        batch = torch.stack(tnsr)
        labels = torch.zeros(1)

        self.batches_generated += 1
        self.idx += 1

        return batch, labels


class AttributesValuesWithDistractorsDataset(AttributesValuesDataset):
    """
    >>> d1 = AttributesValuesWithDistractorsDataset(3, 4, 20, 5, 1)
    >>> epoch1 = [batch for batch in d1]
    >>> len(epoch1)
    4
    >>> epoch1[0][0].shape
    torch.Size([5, 3])
    >>> epoch1[0][1].shape
    torch.Size([5])
    >>> epoch1[0][2].shape
    torch.Size([5, 2, 3])
    """

    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        n_train_samples_per_epoch: int,
        batch_size: int,
        distractors: int = 1,
        seed: int = 111,
    ):
        super().__init__(
            n_attributes, n_values, n_train_samples_per_epoch, batch_size, seed
        )
        self.distractors = distractors

    def __iter__(self):
        return AttributesValuesWithDistractorsIterator(
            self.samples,
            self.batch_size,
            self.n_batches_per_epoch,
            self.distractors,
            self.seed,
        )


class AttributesValuesWithDistractorsIterator(AttributesValuesIterator):
    """
    >>> samples = list(itertools.product(*(range(1, 4) for _ in range(3))))
    >>> it1 = AttributesValuesWithDistractorsIterator(samples, 10, 2, 1, 22)
    >>> it2 = AttributesValuesWithDistractorsIterator(samples, 10, 2, 1, 22)
    >>> l1 = [batch[0] for batch in it1]
    >>> l2 = [batch[0] for batch in it2]
    >>> all([v1.allclose(v2) for v1, v2 in zip(l1, l2)])
    True
    """

    def __init__(
        self,
        samples: Iterable,
        batch_size: int,
        n_batches_per_epoch: int,
        distractors: int = 1,
        seed: int = None,
    ):
        super().__init__(samples, batch_size, n_batches_per_epoch, seed)
        self.distractors = distractors

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.idx >= len(self.data)
            or self.batches_generated >= self.n_batches_per_epoch
        ):
            self.batches_generated = 0
            self.idx = 0
            raise StopIteration()

        if not self.batches_generated:
            self.data = self.random_state.permutation(self.data)

        # fmt: off
        idxs = self.random_state.randint(len(self.data), size=(self.batch_size * (self.distractors+1)))  # noqa: E226
        receiver_input = np.reshape(self.data[idxs], (self.batch_size, self.distractors+1, -1))  # noqa: E226
        labels = self.random_state.choice(self.distractors+1, size=self.batch_size)  # noqa: E226
        # fmt: on
        target = receiver_input[np.arange(self.batch_size), labels]  # noqa: E226

        self.batches_generated += 1
        self.idx += 1

        target = torch.from_numpy(target).float()
        labels = torch.from_numpy(labels).long()
        receiver_input = torch.from_numpy(receiver_input).float()
        return target, labels, receiver_input
