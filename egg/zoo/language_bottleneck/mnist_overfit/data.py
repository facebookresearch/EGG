# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def corrupt_labels_(dataset, p_corrupt, seed):
    random_state = np.random.RandomState(seed)
    original_labels = dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels

    mask = random_state.binomial(n=1, size=original_labels.size(), p=p_corrupt)
    mask = torch.from_numpy(mask)
    random_indexes = random_state.permutation(original_labels.size(0))
    random_indexes = torch.from_numpy(random_indexes)

    random_labels = original_labels[random_indexes]

    new_labels = original_labels * (1 - mask) + mask * random_labels

    if hasattr(dataset, 'targets'):
        dataset.targets = new_labels
    else:
        dataset.train_labels = new_labels

    if hasattr(dataset, 'targets'):
        dataset.targets = new_labels
    else:
        dataset.train_labels = new_labels

    return dataset

