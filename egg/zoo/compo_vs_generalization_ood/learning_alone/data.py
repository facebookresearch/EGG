# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import DataLoader

from egg.zoo.compo_vs_generalization_ood.train import get_data as get_data_splits

EOS_TOKEN = 0
PAD_TOKEN = -100


def remap_batch(batch, mapping, device, swap_xy=False):
    """
    given a mapping, e.g.: 0 > 3, 1 > 2, 2 > 0, 3 > 1
    and a batch, e.g.:
                        0, 1
                        3, 3
                        2, 2

    this returns the original batch and a translation (applying the mapping and adding 1);
    each even index in the translation is duplicated:
                        4, 4, 3, 0, -100
                        2, 2, 2, 2, 0
                        1, 1, 0, -100, -100
    (0: EOS symbol, -100: PAD symbol)
    """
    b = torch.stack(batch).to(device)  # List[examples] -> torch.LongTensor
    new_b = mapping[torch.stack(batch)] + 1
    rows = []
    for row in new_b:
        new_row = []
        for val in row:
            new_row.append(val.item())
            if val % 2 == 0:
                new_row.append(val.item())
        rows.append(
            new_row + [EOS_TOKEN] + [PAD_TOKEN] * (2 * b.shape[1] - len(new_row))
        )
    new_b = torch.LongTensor(rows).to(device)
    return (b, new_b) if not swap_xy else (new_b, b)  # sender vs receiver test


def datasetify(examples, opts, mapping, swap=False, shuffle=False):  # list of tuples
    data = torch.LongTensor(examples)
    return DataLoader(
        data,
        batch_size=opts.batch_size,
        collate_fn=lambda x: remap_batch(x, mapping, opts.device, swap),
        shuffle=shuffle,
    )


def get_data(opts):
    swap = opts.archpart == "receiver"
    _, train, uniform_holdout, generalization_holdout = get_data_splits(opts)
    mapping = torch.randperm(opts.n_values)
    return {
        "train": datasetify(train, opts, mapping, swap, shuffle=True),
        "test_unif": datasetify(uniform_holdout, opts, mapping, swap),
        "test_ood": datasetify(generalization_holdout, opts, mapping, swap),
    }
