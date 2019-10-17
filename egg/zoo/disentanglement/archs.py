# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x



class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_inputs, n_hidden, bias=False)
        self.fc = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        x = self.emb(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x


class PositionalSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, max_len):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len

        log = 0
        k = 1

        while k < n_values:
            k *= (vocab_size - 1)
            log += 1

        assert log * n_attributes < max_len

        self.mapping = nn.Embedding(n_values, log)
        torch.nn.init.zeros_(self.mapping.weight)

        for i in range(n_values):
            value = i
            for k in range(log):
                self.mapping.weight[i, k] = 1 + value % (vocab_size - 1) # avoid putting zeros!
                value = value // (vocab_size - 1)

        assert (self.mapping.weight < vocab_size).all()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_attributes, self.n_values)
        x = x.argmax(dim=-1).view(batch_size * self.n_attributes)
        with torch.no_grad():
            x = self.mapping(x)
        x = x.view(batch_size, -1).long()
        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)

        return x, zeros, zeros

if __name__ == '__main__':
    mapper = PositionalSender(n_attributes=2, n_values=10, vocab_size=10, max_len=7)
    input = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.]])
    print(mapper(input))