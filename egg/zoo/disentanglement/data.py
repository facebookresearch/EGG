# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import itertools
import random

def enumerate_attribute_value(n_attributes, n_values):
    iters = [range(n_values) for _ in range(n_attributes)]

    return list(itertools.product(*iters))

def one_hotify(data, n_attributes, n_values):
    for config in data:
        z = torch.zeros((n_attributes, n_values))
        for i in range(n_attributes):
            z[i, config[i]] = 1
        yield z.view(-1)


def split_by_attribute_value(dataset, attribute, value):
    with_av = [x for x in dataset if x[attribute] == value]
    without_av = [x for x in dataset if x[attribute] != value]

    return with_av, without_av


def split_train_test(dataset, p_hold_out=0.1, random_seed=7):
    import numpy as np

    assert p_hold_out > 0
    random_state = np.random.RandomState(seed=random_seed)

    n = len(dataset)
    permutation = random_state.permutation(n)

    n_test = int(p_hold_out * n)

    test = [dataset[i] for i in permutation[:n_test]]
    train = [dataset[i] for i in permutation[n_test:]]
    assert train and test

    assert len(train) + len(test) == len(dataset) 
    return train, test


class ScaledDataset:
    def __init__(self, examples, scaling_factor=1):
        self.examples = examples
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.examples) * self.scaling_factor

    def __getitem__(self, k):
        k = k % len(self.examples)
        return self.examples[k], torch.zeros(1)


if __name__ == '__main__':
    dataset = enumerate_attribute_value(n_attributes=2, n_values=10)
    train, test = split_train_test(dataset, 0.5)
    print(len(train), len(test), len(dataset))
    print([x[0] for x in [train, test, dataset]])