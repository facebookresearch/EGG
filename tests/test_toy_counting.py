# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import torch

import egg.core as core
from egg.core import Interaction

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())


class ToyDataset:
    def __init__(self):
        data = []

        for i in range(2 ** 8):
            s = [int(x) for x in "{0:08b}".format(i)]
            data.append(s)
        self.data = torch.tensor(data).float()
        self.y = self.data.sum(dim=1)

    def __iter__(self):
        return iter([[self.data, self.y]])


class ToyAgent(torch.nn.Module):
    def __init__(self):
        super(ToyAgent, self).__init__()
        self.fc1 = torch.nn.Linear(8, 1, bias=False)

    def forward(self, x, aux_input=None):
        x = self.fc1(x)
        return x


class ToyGame(torch.nn.Module):
    def __init__(self, agent):
        super(ToyGame, self).__init__()

        self.agent = agent
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y, z=None, aux_input=None):
        output = self.agent(x)
        output = output.squeeze(1)
        loss = self.criterion(output, y)
        return loss, Interaction.empty()


def test_toy_counting_gradient():
    core.init()

    agent = ToyAgent()
    game = ToyGame(agent)
    optimizer = core.build_optimizer(agent.parameters())

    d = ToyDataset()
    trainer = core.Trainer(game, optimizer, train_data=d, validation_data=None)
    trainer.train(10000)

    are_close = torch.allclose(
        agent.fc1.weight, torch.ones_like(agent.fc1.weight), rtol=0.05
    )
    assert are_close, agent.fc1.weight
