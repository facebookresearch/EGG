# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import sys
from pathlib import Path

import torch

import egg.core as core
from egg.core import Callback, LoggingStrategy

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())

batch_size = 16
data_size = 10

BATCH_X = torch.zeros((batch_size, data_size))
BATCH_Y = torch.ones(batch_size).long()


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        for batch in range(self.batch_size):
            batch_x = BATCH_X[batch]
            batch_y = BATCH_Y[batch]

            yield (batch_x, batch_y)


class AssertCallback(Callback):
    def __init__(self, mode):

        if mode == 1:
            self.on_epoch_end = self.on_epoch_end_mixed

        elif mode == 3:

            self.on_epoch_end = self.on_epoch_end_mixed

        elif mode == 6:
            self.on_epoch_end = self.on_epoch_end_logging_step

        elif mode == 7:
            self.on_epoch_end = self.on_epoch_end_logging_store

    def on_epoch_end_logging_store(
        self, loss: float, logs: core.Interaction, epoch: int
    ):

        assert logs.sender_input is None
        assert logs.labels is None
        assert logs.message is None

    def on_epoch_end_logging_step(
        self, loss: float, logs: core.Interaction, epoch: int
    ):

        bs = batch_size // 2

        assert logs.sender_input.shape == torch.Size([bs, data_size])
        if logs.receiver_input is not None:
            if isinstance(logs.receiver_input, torch.Tensor):
                assert logs.receiver_input.shape == torch.Size([bs])
            else:
                assert len(logs.receiver_input) == bs

    def on_epoch_end_mixed(self, loss: float, logs: core.Interaction, epoch: int):

        assert logs.sender_input.shape == torch.Size([batch_size, data_size])
        if logs.receiver_input is not None:
            if isinstance(logs.receiver_input, torch.Tensor):
                assert logs.receiver_input.shape == torch.Size([batch_size])
            else:
                assert len(logs.receiver_input) == batch_size

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        pass


class MockGame(torch.nn.Module):
    def __init__(self, mode=-1):
        super(MockGame, self).__init__()
        self.param = torch.nn.Parameter(torch.Tensor([0]))
        if mode == 0:
            # list check
            self.forward = self.forward_list
        elif mode == 1:
            self.forward = self.forward_mixed_types
        elif mode == 2:
            self.forward = self.forward_empty
        elif mode == 3:
            self.forward = self.forward_mixed_none
        elif mode == 4:
            self.forward = self.forward_mixed_dimension_list
        elif mode == 5:
            self.forward = self.forward_mixed_data_types

    def forward(self, *args, **kwargs):
        """
        Test for interaction made of list only
        """
        interaction = core.Interaction(
            sender_input=args[0],
            receiver_input=float(args[1]),
            labels=list(range(10)),
            message=list(range(10)),
            receiver_output=list(range(10)),
            message_length=list(range(10)),
            aux_input=dict(
                prov=list(range(10)),
            ),
            aux=dict(
                prov=list(range(10)),
            ),
        )
        return self.param, interaction

    def forward_list(self, *args, **kwargs):
        """
        Test for interaction made of list only
        """
        interaction = core.Interaction(
            sender_input=args[0].tolist(),
            receiver_input=args[1].tolist(),
            labels=list(range(10)),
            message=list(range(10)),
            receiver_output=list(range(10)),
            message_length=list(range(10)),
            aux_input=dict(
                prov=list(range(10)),
            ),
            aux=dict(
                prov=list(range(10)),
            ),
        )
        return self.param, interaction

    def forward_mixed_types(self, *args, **kwargs):
        """
        Test for interaction made of list and torch tensors
        """
        interaction = core.Interaction(
            sender_input=args[0],
            receiver_input=float(args[1]),
            labels=list(range(10)),
            message=list(range(10)),
            receiver_output=list(range(10)),
            message_length=list(range(10)),
            aux_input=dict(
                prov=list(range(10)),
            ),
            aux=dict(
                prov=list(range(10)),
            ),
        )
        return self.param, interaction

    def forward_empty(self, *args, **kwargs):
        """
        Test for interaction made of list and torch tensors
        """

        if random.uniform(0, 1) > 0.5:

            interaction = core.Interaction(
                sender_input=args[0],
                receiver_input=float(args[1]),
                labels=list(range(10)),
                message=list(range(10)),
                receiver_output=list(range(10)),
                message_length=list(range(10)),
                aux_input=dict(
                    prov=list(range(10)),
                ),
                aux=dict(
                    prov=list(range(10)),
                ),
            )
        else:
            interaction = core.Interaction.empty()
        return self.param, interaction

    def forward_mixed_none(self, *args, **kwargs):
        """
        Test for interaction made of list and torch tensors
        """

        interaction = core.Interaction(
            sender_input=args[0],
            receiver_input=None,
            labels=list(range(10)),
            message=None,
            receiver_output=None,
            message_length=list(range(10)),
            aux_input=dict(
                prov=list(range(10)),
            ),
            aux={},
        )
        return self.param, interaction

    def forward_mixed_dimension_list(self, *args, **kwargs):
        """
        Test for interaction made of list and torch tensors
        """
        sender_input = args[0].tolist()
        receiver_input = args[0].tolist()

        if random.uniform(0, 1) > 0.5:
            sender_input = sender_input[: batch_size // 2]
            receiver_input = receiver_input[: batch_size // 2]

        interaction = core.Interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=None,
            message=None,
            receiver_output=None,
            message_length=None,
            aux_input=None,
            aux={},
        )
        return self.param, interaction

    def forward_mixed_data_types(self, *args, **kwargs):
        """
        Test for interaction made of list and torch tensors
        """
        sender_input = args[0]
        receiver_input = args[1]

        if random.uniform(0, 1) > 0.5:
            sender_input = sender_input.tolist()
            receiver_input = receiver_input.tolist()

        interaction = core.Interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=None,
            message=None,
            receiver_output=None,
            message_length=None,
            aux_input=None,
            aux={},
        )
        return self.param, interaction


def test_list_interaction():
    core.init(params=[])
    mode = 0
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )
    trainer.train(1)


def test_mixed_interaction():
    core.init(params=[])
    mode = 1
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )
    trainer.train(1)


def test_empty_interaction():
    core.init(params=[])
    mode = 2
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )
    trainer.train(1)


def test_mixed_none_interaction():
    core.init(params=[])
    mode = 3
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )
    trainer.train(1)


def test_mixed_dimension_list_interaction():
    core.init(params=[])
    mode = 4
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )
    passed = False
    try:
        trainer.train(1)
    except RuntimeError:
        passed = True

    assert passed, "Mixed dimension test did not pass"


def test_mixed_data_type_interaction():
    core.init(params=[])
    mode = 5
    game = MockGame(mode=mode)
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        callbacks=[AssertCallback(mode=mode)],
    )

    passed = False
    try:
        trainer.train(1)
    except RuntimeError:
        passed = True

    assert passed, "Mixed dimension test did not pass"


def test_logging_step():
    core.init(params=[])
    mode = 6
    game = MockGame()
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    custom_logging = LoggingStrategy(logging_step=2)

    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        train_logging_strategy=custom_logging,
        callbacks=[AssertCallback(mode=mode)],
    )

    trainer.train(1)


def test_logging_store():
    core.init(params=[])
    mode = 7
    game = MockGame()
    optimizer = torch.optim.Adam(game.parameters())

    data = Dataset(batch_size=batch_size)
    custom_logging = LoggingStrategy(
        store_sender_input=False,
        store_labels=False,
        store_message=False,
    )

    trainer = core.Trainer(
        game,
        optimizer,
        train_data=data,
        train_logging_strategy=custom_logging,
        callbacks=[AssertCallback(mode=mode)],
    )

    trainer.train(1)
