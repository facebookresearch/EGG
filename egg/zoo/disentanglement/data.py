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


def split_train_test(dataset, p_hold_out=0):
    assert p_hold_out >= 0

    train = dataset
    test = []

    if p_hold_out > 0:
        random.shuffle(train)
        n_test = int(p_hold_out * len(train))
        train, test = train[n_test:], train[:n_test]

        assert train and test

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

    holdout_a1, rest = split_by_attribute_value(dataset, attribute=0, value=9)
    holdout_a2, rest = split_by_attribute_value(dataset, attribute=1, value=9)

    train = rest

    print(len(dataset), len(holdout_a1), len(holdout_a2), len(rest))
