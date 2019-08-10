# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import torch.nn as nn
import egg.core as core
import torch.nn.functional as F


class Sender(nn.Module):
    def __init__(self, n_colors, vocab_size):
        super(Sender, self).__init__()
        self.emb = nn.Embedding(n_colors, 50)
        self.fc = nn.Linear(50, vocab_size)

    def forward(self, x):
        x = x[:, 0:1] # only color-id at the moment
        x = self.emb(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)


class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()

    def forward(self, x, _input):
        return x.squeeze(1)

