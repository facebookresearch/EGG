# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import egg.core as core

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())

BATCH_X = torch.eye(8)
BATCH_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).long()


class Dataset:
    def __iter__(self):
        return iter([(BATCH_X, BATCH_Y)])


class ToyAgent(torch.nn.Module):
    def __init__(self):
        super(ToyAgent, self).__init__()
        self.fc1 = torch.nn.Linear(8, 2, bias=False)

    def forward(self, x, _aux_input=None):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Receiver(torch.nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()

    def forward(self, x, _input=None, _aux_input=None):
        return x


def test_toy_agent_gs():
    core.init()
    agent = core.GumbelSoftmaxWrapper(ToyAgent())

    agent.eval()
    output = agent(BATCH_X)
    assert output.size() == torch.Size((8, 2))
    assert (output > 0).sum() == 8

    agent.train()
    agent.temperature = 10.0
    output = agent(BATCH_X, {})
    assert output.size() == torch.Size((8, 2))
    assert (output > 0).sum() == 16

    agent.temperature = 0.5

    optimizer = torch.optim.Adam(agent.parameters())

    for _ in range(1000):
        optimizer.zero_grad()
        out = agent(BATCH_X, {})
        loss = F.cross_entropy(out, BATCH_Y)
        loss.backward()

        optimizer.step()

    assert (agent.agent.fc1.weight.t().argmax(dim=1) == BATCH_Y).all()


def test_game_gs():
    core.init()
    sender = core.GumbelSoftmaxWrapper(ToyAgent())
    receiver = Receiver()
    loss = lambda sender_input, message, receiver_input, receiver_output, labels, aux_input: (
        F.cross_entropy(receiver_output, labels),
        {},
    )

    game = core.SymbolGameGS(sender, receiver, loss)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset()
    trainer = core.Trainer(game, optimizer, train_data=data, validation_data=None)
    trainer.train(1000)

    assert (sender.agent.fc1.weight.t().argmax(dim=1).cpu() == BATCH_Y).all()


def test_toy_agent_reinforce():
    core.init()
    agent = core.ReinforceWrapper(ToyAgent())

    optimizer = torch.optim.Adam(agent.parameters())

    for _ in range(1000):
        optimizer.zero_grad()
        output, log_prob, entropy = agent(BATCH_X, {})
        loss = -((output == BATCH_Y).float() * log_prob).mean()
        loss.backward()

        optimizer.step()

    assert (agent.agent.fc1.weight.t().argmax(dim=1).cpu() == BATCH_Y).all()


def test_game_reinforce():
    core.init()
    sender = core.ReinforceWrapper(ToyAgent())
    receiver = core.ReinforceDeterministicWrapper(Receiver())

    loss = lambda sender_input, message, receiver_input, receiver_output, labels, aux_input: (
        -(receiver_output == labels).float(),
        {},
    )

    game = core.SymbolGameReinforce(
        sender, receiver, loss, sender_entropy_coeff=1e-1, receiver_entropy_coeff=0.0
    )
    optimizer = torch.optim.Adagrad(game.parameters(), lr=1e-1)

    data = Dataset()
    trainer = core.Trainer(game, optimizer, train_data=data, validation_data=None)
    trainer.train(5000)

    assert (sender.agent.fc1.weight.t().argmax(dim=1).cpu() == BATCH_Y).all(), str(
        sender.agent.fc1.weight
    )


def test_symbol_wrapper():
    core.init()

    receiver = core.SymbolReceiverWrapper(Receiver(), vocab_size=15, agent_input_size=5)

    # when trained with REINFORCE, the message would be encoded as long ids
    message_rf = torch.randint(high=15, size=(16,)).long()
    output_rf = receiver(message_rf)

    assert output_rf.size() == torch.Size((16, 5))

    # when trained with Gumbel-Softmax, the message would be encoded as one-hots
    message_gs = torch.zeros((16, 15))
    message_gs.scatter_(
        1, message_rf.unsqueeze(1), 1.0
    )  # same message, one-hot-encoded
    output_gs = receiver(message_gs)

    assert output_rf.eq(output_gs).all().item() == 1
