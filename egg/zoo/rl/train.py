# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import wandb

import torch

import egg.core as core
from egg.zoo.rl.dataloader import get_loader
from egg.zoo.rl.models import build_game
from egg.zoo.rl.utils import get_opts


# example_images.append(wandb.Image(
#    data[0],
#    caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])
#    )
# )
# wandb.log({
#    "Examples": example_images,
#    "Test Accuracy": 100. * correct / len(test_loader.dataset),
#    "Test Loss": test_loss}
# )


def main():
    wandb.init(project="language-as-rl")
    opts = get_opts()
    wandb.config.update(opts)

    train_loader = get_loader()

    game = build_game()

    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.sender_lr},
        {'params': game.receiver.parameters(), 'lr': opts.receiver_lr}
    ])

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
    )
    trainer.train(n_epochs=opts.n_epochs)

    wandb.watch(game)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
