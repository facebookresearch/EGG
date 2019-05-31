# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import torch.utils.data as data
import torch.nn.parallel
import os
import torch
import numpy as np


class _DataIterator:
    def __init__(self, max_n, n_batches_per_epoch, batch_size, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.max_n = max_n
        self.batch_size = batch_size

        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)
        assert batch_size % 2 == 0

    def __iter__(self):
        return self

    def generate_positive_examples(self, n_examples):
        generated_n = self.random_state.randint(1, self.max_n, n_examples)

        batch_data = np.zeros((n_examples, 2 * self.max_n + 1), dtype=np.int)

        for i in range(n_examples):
            n = generated_n[i]
            batch_data[i, :n] = 1
            batch_data[i, n:2*n] = 2

            assert (batch_data[i, :] == 1).sum() == (batch_data[i, :] == 2).sum()
        return torch.from_numpy(batch_data), torch.from_numpy(2 * generated_n + 1)

    def generate_negative_examples(self, n_examples):
        batch_data = np.zeros((n_examples, self.max_n * 2 + 1), dtype=np.int)
        lengths = np.zeros(n_examples, dtype=np.int)

        for i in range(n_examples):
            while True:
                n1, n2 = self.random_state.randint(0, self.max_n, 2)

                if n1 != n2 and n1 + n2 <= 2 * self.max_n:
                    break

            batch_data[i, :n1] = 1
            batch_data[i, n1:n1 + n2] = 2
            lengths[i] = n1 + n2 + 1

            assert (batch_data[i, :] == 1).sum() != (batch_data[i, :] == 2).sum()
        return torch.from_numpy(batch_data), torch.from_numpy(lengths)

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        positive_seq, positive_len = self.generate_positive_examples(self.batch_size // 2)
        negative_seq, negative_len = self.generate_negative_examples(self.batch_size // 2)

        examples = torch.cat([positive_seq, negative_seq], dim=0)
        lengths = torch.cat([positive_len, negative_len])
        labels = torch.zeros_like(lengths)
        labels[:self.batch_size // 2] = 1

        _, rearrange = torch.sort(lengths, descending=True)
        examples = torch.index_select(examples, 0, rearrange)
        lengths = torch.index_select(lengths, 0, rearrange)
        labels = torch.index_select(labels, 0, rearrange)

        self.batches_generated += 1
        return (examples, lengths), labels


class SequenceLoader(torch.utils.data.DataLoader):
    def __init__(self, max_n, batches_per_epoch, batch_size, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.max_n = max_n

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _DataIterator(max_n=self.max_n,
                             n_batches_per_epoch=self.batches_per_epoch,
                             batch_size=self.batch_size, seed=seed)

