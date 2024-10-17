# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class Receiver(nn.Module):
    def __init__(self, n_features, linear_units):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()
