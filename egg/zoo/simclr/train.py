# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core
from egg.zoo.simclr.utils import get_dataloader
from egg.zoo.simclr.games import build_game


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CIFAR10",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset download location",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Model name for the encoder",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image size"
    )
    parser.add_argument(
        "--ntxent_tau",
        type=float,
        default=0.1,
        help="Temperature for NT XEnt loss",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=64,
        help="Projection head's dimension"
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
        default=128,
        help="Receiver output size"
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

    train_loader = get_dataloader(opts)

    simclr_game = build_game(opts)

    optimizer = core.build_optimizer(simclr_game.parameters())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)]
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
