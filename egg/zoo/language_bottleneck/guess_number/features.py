# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np


def sender_receiver_examples(examples, n_bits, bits_s, bits_r):
    sender_examples = np.copy(examples)
    sender_examples[:, bits_s:] = 0
    sender_examples = torch.from_numpy(sender_examples)

    receiver_examples = np.copy(examples)
    receiver_examples[:, :n_bits - bits_r] = 0
    receiver_examples = torch.from_numpy(receiver_examples)

    examples = torch.from_numpy(examples)

    return sender_examples, examples, receiver_examples


class _OneHotIterator:
    """
    >>> it = _OneHotIterator(n_bits=8, bits_s=4, bits_r=4, n_batches_per_epoch=1, batch_size=128)
    >>> batch = list(it)[0]
    >>> s, l, r = batch
    >>> ((s + r) == l).all().item()
    1
    >>> it = _OneHotIterator(n_bits=8, bits_s=5, bits_r=5, n_batches_per_epoch=1, batch_size=128)
    >>> batch = list(it)[0]
    >>> ((s + r).clamp(0, 1) == l).all().item()
    1
    >>> it = _OneHotIterator(n_bits=8, bits_s=8, bits_r=8, n_batches_per_epoch=1, batch_size=128)
    >>> batch = list(it)[0]
    >>> s, l, r = batch
    >>> (s == r).all().item()
    1
    >>> it = _OneHotIterator(n_bits=8, bits_s=8, bits_r=1, n_batches_per_epoch=1, batch_size=128)
    >>> batch = list(it)[0]
    >>> s, l, r = batch
    >>> (r[:, -1] > 0).any().item()
    1
    >>> (r[:, :-1] == 0).all().item()
    1
    """
    def __init__(self, n_bits, bits_s, bits_r, n_batches_per_epoch, batch_size, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.n_bits = n_bits
        self.bits_s = bits_s
        self.bits_r = bits_r
        self.batch_size = batch_size

        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        examples = self.random_state.randint(low=0, high=2, size=(self.batch_size, self.n_bits))

        sender_examples, examples, receiver_examples = \
            sender_receiver_examples(examples, self.n_bits, self.bits_s, self.bits_r)

        self.batches_generated += 1
        return sender_examples, examples, receiver_examples


class OneHotLoader(torch.utils.data.DataLoader):
    def __init__(self, n_bits, bits_s, bits_r, batches_per_epoch, batch_size, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.n_bits = n_bits
        self.bits_r = bits_r
        self.bits_s = bits_s
        self.batch_size = batch_size

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _OneHotIterator(n_bits=self.n_bits, bits_s=self.bits_s, bits_r=self.bits_r,
                               n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, seed=seed)


class UniformLoader(torch.utils.data.DataLoader):
    def __init__(self, n_bits, bits_s, bits_r):

        batch_size = 2**(n_bits)
        numbers = np.array(range(batch_size))

        examples = np.zeros((batch_size, n_bits), dtype=np.int)

        for i in range(n_bits):
            examples[:, i] = np.bitwise_and(numbers, 2 ** i) > 0

        sender_examples, examples, receiver_examples = \
            sender_receiver_examples(examples, n_bits, bits_s, bits_r)

        self.batch = sender_examples, examples, receiver_examples

    def __iter__(self):
        return iter([self.batch])

