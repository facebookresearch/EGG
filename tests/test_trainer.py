# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import shutil
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())

import torch
from torch.nn import functional as F

import egg.core as core


BATCH_X = torch.eye(8)
BATCH_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).long()


class Dataset:
    def __iter__(self):
        return iter([(BATCH_X, BATCH_Y)])


class Receiver(torch.nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()

    def forward(self, x, _input):
        return x


class ToyAgent(torch.nn.Module):
    def __init__(self):
        super(ToyAgent, self).__init__()
        self.fc1 = torch.nn.Linear(8, 2, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class MockGame(torch.nn.Module):
    def __init__(self):
        super(MockGame, self).__init__()
        self.param = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, *args, **kwargs):
        return self.param, {'acc': 1}


def test_temperature_updater_callback():
    core.init()
    sender = core.GumbelSoftmaxWrapper(ToyAgent(), temperature=1)
    receiver = Receiver()
    loss = lambda sender_input, message, receiver_input, receiver_output, labels: \
        (F.cross_entropy(receiver_output, labels), {})

    game = core.SymbolGameGS(sender, receiver, loss)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset()
    trainer = core.Trainer(game, optimizer, train_data=data, validation_data=None,
                           callbacks=[core.TemperatureUpdater(agent=sender, decay=0.9)])
    trainer.train(1)
    assert sender.temperature == 0.9


def test_snapshoting():
    CHECKPOINT_PATH = Path('./test_checkpoints')

    core.init()
    sender = core.GumbelSoftmaxWrapper(ToyAgent(), temperature=1)
    receiver = Receiver()
    loss = lambda sender_input, message, receiver_input, receiver_output, labels: \
        (F.cross_entropy(receiver_output, labels), {})

    game = core.SymbolGameGS(sender, receiver, loss)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset()
    trainer = core.Trainer(game, optimizer, train_data=data, validation_data=None,
                           callbacks=[core.CheckpointSaver(checkpoint_path=CHECKPOINT_PATH)])
    trainer.train(2)
    assert (CHECKPOINT_PATH / Path('0.tar')).exists() and (CHECKPOINT_PATH / Path('1.tar')).exists()
    assert (CHECKPOINT_PATH / Path('final.tar')).exists()
    del trainer
    trainer = core.Trainer(game, optimizer, train_data=data)  # Re-instantiate trainer
    trainer.load_from_latest(CHECKPOINT_PATH)
    assert trainer.starting_epoch == 2
    trainer.train(3)
    shutil.rmtree(CHECKPOINT_PATH)  # Clean-up


def test_early_stopping():
    game, data = MockGame(), Dataset()
    early_stopper = core.EarlyStopperAccuracy(threshold=0.9)
    trainer = core.Trainer(game=game, optimizer=torch.optim.Adam(game.parameters()), train_data=data,
                           validation_data=data, callbacks=[early_stopper])
    trainer.train(1)
    assert trainer.should_stop
