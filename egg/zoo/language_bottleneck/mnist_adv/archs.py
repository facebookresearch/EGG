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
    def __init__(self, vocab_size, linear_channel, softmax_channel):
        super(Sender, self).__init__()

        self.vision = LeNet()
        self.fc = nn.Linear(400, vocab_size)
        self.linear_channel = linear_channel
        self.softmax_channel = softmax_channel

    def forward(self, x, _aux_input):
        x = self.vision(x)
        x = self.fc(x)

        if not self.linear_channel and not self.softmax_channel:
            return F.log_softmax(x, dim=1)
        elif self.linear_channel:
            return x
        elif self.softmax_channel:
            return x.softmax(dim=1)


class Receiver(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super(Receiver, self).__init__()
        self.message_inp = core.RelaxedEmbedding(vocab_size, 400)
        self.fc = nn.Linear(400, n_classes)

    def forward(self, message, input, _aux_input):
        x = self.message_inp(message)
        x = self.fc(x)

        return torch.log_softmax(x, dim=1)
