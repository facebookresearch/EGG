# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core
from egg.core import Callback, EarlyStopperAccuracy, Interaction
from egg.zoo.simclr.utils import get_dataloader
from egg.zoo.simclr.games import build_game


def get_opts(params):
    parser = argparse.ArgumentParser()

    # Data opts
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imagenet"],
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset location",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Workers used in the dataloader"
    )

    # Vision module opts
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )
    parser.add_argument(
        "--shared_vision",
        default=False,
        action="store_true",
        help="If set, Sender and Receiver will share the vision encoder"
    )

    # Loss opts
    parser.add_argument(
        "--ntxent_tau",
        type=float,
        default=0.1,
        help="Temperature for NT XEnt loss",
    )

    # Arch opts
    parser.add_argument(
        "--vision_projection_dim",
        type=int,
        default=64,
        help="Projection head's dimension for image features"
    )
    parser.add_argument(
        "--sender_output_size",
        type=int,
        default=128,
        help="Sender output size"
    )
    parser.add_argument(
        "--receiver_output_size",
        type=int,
        default=256,
        help="Receiver output size"
    )

    # Misc opts
    parser.add_argument(
        "--early_stopping_thr",
        type=float, default=0.99,
        help="Early stopping threshold on accuracy (defautl: 0.99)"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled"
    )

    opts = core.init(arg_parser=parser, params=params)
    return opts


def main(params):
    opts = get_opts(params=params)
    if opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        opts.dataset_name,
        opts.dataset_dir,
        opts.image_size,
        opts.batch_size,
        opts.num_workers
    )

    simclr_game = build_game(opts)

    optimizer = core.build_optimizer(simclr_game.parameters())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            EarlyStopperAccuracy(opts.early_stopping_thr, validation=False),
            BestStatsTracker()
        ]
    )
    trainer.train(n_epochs=opts.n_epochs)


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_acc = -1.
        self.best_loss = float("inf")
        self.best_epoch = -1

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_acc:
            self.best_acc = logs.aux["acc"].mean().item()
            self.best_epoch = epoch
            self.best_loss = _loss

    def on_train_end(self):
        print(f"BEST: epoch {self.best_epoch}, acc: {self.best_acc}, loss: {self.best_loss}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
