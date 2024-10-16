# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input, _aux_input):
        return self.output(x)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input):
        x = self.fc1(x)
        return x
