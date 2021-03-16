# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data")
    group.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagenet"],
        help="Dataset name",
    )
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset location",
    )
    group.add_argument(
        "--validation_dataset_dir",
        type=str,
        default="./val_data",
        help="Dataset location",
    )
    group.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image size"
    )
    group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers used in the dataloader"
    )


def get_rf_opts(parser):
    group = parser.add_argument_group("reinforce")
    # sender opts
    group.add_argument(
        "--recurrent_cell",
        type=str,
        default="rnn",
        choices=["rnn", "lstm", "gru"],
        help="Type of the cell used for Sender and Receiver {rnn, gru, lstm} (default: rnn)"
    )
    group.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=0.1,
        help="Entropy regularisation coeff for Sender (default: 0.1)"
    )
    group.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)"
    )
    group.add_argument(
        "--sender_rnn_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of sender. Only in reinforce (default: 1)"
    )
    # receiver opts
    group.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)"
    )
    group.add_argument(
        "--receiver_rnn_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)"
    )
    group.add_argument(
        "--receiver_rnn_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of receiver. Only in reinforce (default: 1)"
    )


def get_gs_opts(parser):
    pass


def get_continuous_opts(parser):
    group = parser.add_argument_group("continuous")
    group.add_argument(
        "--sender_output_size",
        type=int,
        default=128,
        help="Sender output size and message dimension in continuous communication"
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision_module")
    group.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )
    group.add_argument(
        "--shared_vision",
        default=False,
        action="store_true",
        help="If set, Sender and Receiver will share the vision encoder"
    )
    group.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision modules will be used"
    )
    group.add_argument(
        "--use_augmentations",
        action="store_true",
        default=False
    )


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game_arch")
    group.add_argument(
        "--projection_dim",
        type=int,
        default=64,
        help="Projection head's dimension for image features. If <= 0 image features won't be linearly projected"
    )
    group.add_argument(
        "--receiver_output_size",
        type=int,
        default=256,
        help="Receiver output size"
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntxent_tau",
        type=float,
        default=0.1,
        help="Temperature for NT XEnt loss",
    )

    parser.add_argument(
        "--communication_channel",
        type=str,
        default="continuous",
        choices=["continuous", "rf", "gs"],
        help="Type of channel used by the sender (default: continous)",
    )

    parser.add_argument(
        "--early_stopping_thr",
        type=float, default=0.9999,
        help="Early stopping threshold on accuracy (defautl: 0.9999)"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled"
    )

    get_data_opts(parser)
    get_rf_opts(parser)
    get_gs_opts(parser)
    get_continuous_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
