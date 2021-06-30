# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class Receiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(Receiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits, _auxt_input=None):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.sigmoid()


class ReinforcedReceiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits, _aux_input=None):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        probs = x.sigmoid()

        distr = Bernoulli(probs=probs)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = (probs > 0.5).float()
        log_prob = distr.log_prob(sample).sum(dim=1)
        return sample, log_prob, entropy


class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, bits, _aux_input=None):
        x = self.emb(bits.float())
        x = F.leaky_relu(x)
        message = self.fc(x)

        return message
