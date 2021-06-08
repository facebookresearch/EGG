# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn

from egg.zoo.emcom_as_ssl.archs import (
    Game,
    Receiver,
    Sender,
)
from egg.zoo.emcom_as_ssl.losses import get_loss


def get_optimizer_and_scheduler(
    game: nn.Module,
    lr: float
) -> Tuple[
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler._LRSchedule]
]:
    pass


def get_game(opts: argparse.Namespace) -> nn.Module:
    loss = get_loss()

    sender = Sender()
    receiver = Receiver()
    game = Game(sender, receiver, loss)
    if opts.distributed_context.is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
