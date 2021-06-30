# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_dim, inner_layers=-1):
        super(Receiver, self).__init__()
        if inner_layers == -1:
            self.net = nn.Linear(n_hidden, n_dim)
        else:
            l = [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]

            for _ in range(inner_layers):
                l += [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]
            l.append(nn.Linear(n_hidden, n_dim))

            self.net = nn.Sequential(*l)

    def forward(self, x, input, aux_input):
        x = self.net(x)
        return x


class IdentitySender(nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, x, aux_input):
        message = x
        assert message.size(1) == 2

        zeros = torch.zeros(message.size(0), message.size(1), device=x.device)
        return message + 1, zeros, zeros


class RotatedSender(nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, x, aux_input):
        message = torch.zeros_like(x).long()

        if x.size(1) == 2:
            message[:, 0] = (x[:, 0] + x[:, 1]).fmod(self.n_values)
            message[:, 1] = (self.n_values + x[:, 0] - x[:, 1]).fmod(self.n_values)
        else:
            assert False

        assert message.size(1) == 2
        zeros = torch.zeros(message.size(0), message.size(1), device=x.device)
        return message + 1, zeros, zeros


class Lenses(nn.Module):
    def __init__(self, theta):
        super().__init__()

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        self.rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]], requires_grad=False
        )
        self.rotation_matrix = nn.Parameter(self.rotation_matrix)

    def __call__(self, examples):
        with torch.no_grad():
            r = examples.matmul(self.rotation_matrix)
        return r


class CircleSender(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, aux_input):
        assert (x >= -1).all() and (x <= 1).all()
        message = ((x + 1) / 2 * (self.vocab_size - 1)).round().long()
        assert (message < self.vocab_size).all()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return message + 1, zeros, zeros
