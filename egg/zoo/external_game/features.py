# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
import torch
import numpy as np


class CSVDataset(Dataset):
    def __init__(self, path):
        datatypes = [('sender_input', 'S10'), ('label', 'S10')]
        frame = np.loadtxt(path, dtype=datatypes, delimiter=';')
        self.frame = []

        for row in frame:
            sender_input, label = row
            sender_input = torch.tensor(list(map(float, sender_input.split())))
            label = torch.tensor(list(map(int, label.split())))

            self.frame.append((sender_input, label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def get_output_size(self):
        return self.frame[0][1].size(0)

    def get_output_max(self):
        return max(x[1].item() for x in self.frame)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

