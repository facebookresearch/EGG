# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.simclr.data import get_dataloader
from egg.zoo.simclr.games import build_game
from egg.zoo.simclr.utils import BestStatsTracker, VisionModelSaver


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
        default="resnet50",
        choices=["resnet50", "resnet101", "resnet152"],
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

    # Game opts
    parser.add_argument(
        "--communication_channel",
        type=str,
        default="continuous",
        choices=["continuous", "rf"],
        help="Type of channel used by the sender (default: continous)",
    )

    # Arch opts
    parser.add_argument(
        "--vision_projection_dim",
        type=int,
        default=64,
        help="Projection head's dimension for image features"
    )
    parser.add_argument(
        "--receiver_output_size",
        type=int,
        default=256,
        help="Receiver output size"
    )

    # continuous-communication opts
    parser.add_argument(
        "--sender_output_size",
        type=int,
        default=128,
        help="Sender output size and message dimension in continuous communication"
    )

    # rf-based training opts
    # sender opts
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0.1,
        help="Entropy regularisation coeff for Sender (default: 0.1)"
    )
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "gru"],
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)"
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)"
    )
    parser.add_argument(
        "--sender_rnn_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of sender. Only in reinforce (default: 1)"
    )
    # receiver opts
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "gru"],
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)"
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)"
    )
    parser.add_argument(
        "--receiver_rnn_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)"
    )
    parser.add_argument(
        "--receiver_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of receiver. Only in reinforce (default: 1)"
    )

    # Misc opts
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float, default=0.99,
        help="Early stopping threshold on accuracy (defautl: 0.99)"
    )
    parser.add_argument(
        "--store_vision_module",
        action="store_true",
        default=False,
        help="Saving vision_encoder(s)"
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
        dataset_name=opts.dataset_name,
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        use_augmentations=opts.use_augmentations
    )

    simclr_game = build_game(opts)

    optimizer = core.build_optimizer(simclr_game.parameters())

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(opts.early_stopping_thr, validation=False),
        BestStatsTracker()
    ]
    if opts.store_vision_module:
        assert opts.checkpoint_dir, "Missing a path to store vision module"
        callbacks.append(VisionModelSaver(opts.checkpoint_dir))

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
