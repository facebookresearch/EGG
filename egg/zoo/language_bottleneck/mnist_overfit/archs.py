# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

import egg.core as core


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 11 * 50, 400)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        return x


class Sender(nn.Module):
    def __init__(self, vocab_size, deeper, linear_channel, softmax_channel):
        super(Sender, self).__init__()

        if deeper:
            assert not linear_channel
            assert not softmax_channel

        self.deeper = deeper
        self.linear_channel = linear_channel
        self.softmax_channel = softmax_channel

        self.vision = LeNet()
        if self.deeper:
            self.fc1 = nn.Linear(400, 400)
            self.fc2 = nn.Linear(400, vocab_size)
        else:
            self.fc = nn.Linear(400, vocab_size)

    def forward(self, x, _aux_input):
        x = self.vision(x)
        if self.deeper:
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)

        if self.softmax_channel:
            return F.softmax(x, dim=1)
        if self.linear_channel:
            return x
        return F.log_softmax(x, dim=1)


class Receiver(nn.Module):
    def __init__(self, vocab_size, n_classes, deeper):
        super(Receiver, self).__init__()
        self.message_inp = core.RelaxedEmbedding(vocab_size, 400)
        if not deeper:
            self.fc = nn.Linear(400, n_classes)
        else:
            self.fc1 = nn.Linear(400, 400)
            self.fc2 = nn.Linear(400, n_classes)
        self.deeper = deeper

    def forward(self, message, _input, _aux_input):
        x = self.message_inp(message)
        if not self.deeper:
            x = self.fc(x)
        else:
            x = self.fc1(x)
            x = self.fc2(x)

        return torch.log_softmax(x, dim=1)
