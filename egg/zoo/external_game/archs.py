# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributions import Categorical


class ReinforceReceiver(nn.Module):
    def __init__(self, output_size, n_hidden):
        super(ReinforceReceiver, self).__init__()
        self.output = nn.Linear(n_hidden, output_size)

    def forward(self, x, _input, _aux_input):
        logits = self.output(x).log_softmax(dim=1)
        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class Receiver(nn.Module):
    def __init__(self, output_size, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, output_size)

    def forward(self, x, _input, _aux_input):
        return self.output(x)


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input):
        x = self.fc1(x)
        return x
