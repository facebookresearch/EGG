# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input, _aux_input):
        return self.fc(x)


class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)

    def forward(self, x, _aux_input):
        x = self.fc1(x)
        return x


class NonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, _input, _aux_input):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros


class Freezer(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

    def train(self, mode):
        pass

    def forward(self, *input):
        with torch.no_grad():
            r = self.wrapped(*input)
        return r


class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3
