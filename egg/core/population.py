# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import random
import numpy as np
import itertools


class UniformAgentSampler(nn.Module):
    # NB: only a module to facilitate checkpoint persistance
    def __init__(self, senders, receivers, losses):
        super().__init__()

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

    def forward(self):
        return (
            random.choice(self.senders),
            random.choice(self.receivers),
            random.choice(self.losses),
        )


class FullSweepAgentSampler(nn.Module):
    # NB: only a module to facilitate checkpoint persistance
    def __init__(self, senders, receivers, losses):
        super().__init__()

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

        self.senders_order = list(range(len(self.senders)))
        self.receivers_order = list(range(len(self.receivers)))
        self.losses_order = list(range(len(self.losses)))

        self.reset_order()

    def reset_order(self):
        np.random.shuffle(self.senders_order)
        np.random.shuffle(self.receivers_order)
        np.random.shuffle(self.losses_order)

        self.iterator = itertools.product(
            self.senders_order, self.receivers_order, self.losses_order
        )

    def forward(self):
        try:
            sender_idx, recv_idx, loss_idx = next(self.iterator)
        except StopIteration:
            self.reset_order()
            sender_idx, recv_idx, loss_idx = next(self.iterator)
        return self.senders[sender_idx], self.receivers[recv_idx], self.losses[loss_idx]


class PopulationGame(nn.Module):
    def __init__(self, game, agents_loss_sampler):
        super().__init__()

        self.game = game
        self.agents_loss_sampler = agents_loss_sampler

    def forward(self, *args, **kwargs):
        sender, receiver, loss = self.agents_loss_sampler()

        return self.game(sender, receiver, loss, *args, **kwargs)
