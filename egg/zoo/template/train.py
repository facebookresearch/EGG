# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from typing import List

import torch

import egg.core as core
from egg.zoo.template.data import get_dataloader
from egg.zoo.template.game_callbacks import get_callbacks
from egg.zoo.template.games import build_game, build_optimizer_and_scheduler
from egg.zoo.template.utils import get_opts


def main(params: List[str]) -> None:
    begin = (datetime.now() + timedelta(hours=9))
    print(f"| STARTED JOB at {begin}...")

    opts = get_opts(params=params)
    print(f"{opts}\n")
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader()

    game = build_game(opts)

    optimizer, optimizer_scheduler = build_optimizer_and_scheduler(game, opts.lr)

    callbacks = get_callbacks()

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)

    end = (datetime.now() + timedelta(hours=9))  # Using CET timezone

    print(f"| FINISHED JOB at {end}. It took {end - begin}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys
    main(sys.argv[1:])
