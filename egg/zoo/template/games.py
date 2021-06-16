# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from egg.zoo.template.archs import Game, Receiver, Sender
from egg.zoo.template.losses import get_loss


def build_optimizer_and_scheduler(
    game: nn.Module,
    lr: float
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:  # some pytorch schedulers are child classes of object
    pass


def build_game(opts: argparse.Namespace) -> nn.Module:
    loss = get_loss()

    sender = Sender()
    receiver = Receiver()
    game = Game(sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
