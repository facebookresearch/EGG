# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import wandb

import egg.core as core
from egg.zoo.rl.dataloaders import get_loader
from egg.zoo.rl.game import build_game
from egg.zoo.rl.utils import get_opts


def main(params):
    opts = get_opts(params)
    print(opts)
    if opts.pdb:
        breakpoint()
    if opts.wandb:
        wandb.init(project="language-as-rl")
        wandb.config.update(opts)

    train_loader = get_loader(opts)

    game = build_game(opts)

    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.sender_lr},
        {'params': game.receiver.parameters(), 'lr': opts.receiver_lr}
    ])

    print(f"| There are {sum(p.numel() for p in game.parameters() if p.requires_grad)} parameters in the model.")

    callbacks = [core.ConsoleLogger(print_train_loss=True)]
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)

    if opts.wandb:
        wandb.watch(game)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
