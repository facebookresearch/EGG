# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data")
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset location",
    )

    group.add_argument(
        "--dataset_name",
        choices=["cifar10", "imagenet"],
        default="imagenet",
        help="Dataset used for training a model",
    )

    group.add_argument("--image_size", type=int, default=224, help="Image size")

    group.add_argument(
        "--num_workers", type=int, default=4, help="Workers used in the dataloader"
    )
    parser.add_argument(
        "--return_original_image",
        action="store_true",
        default=False,
        help="Dataloader will yield also the non-augmented version of the input images",
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )
    group.add_argument(
        "--gs_temperature_decay",
        type=float,
        default=1.0,
        help="gs temperature update_factor (default: 1.0)",
    )
    group.add_argument(
        "--train_gs_temperature",
        default=False,
        action="store_true",
        help="train gs temperature used in the relaxation layer",
    )
    group.add_argument(
        "--straight_through",
        default=False,
        action="store_true",
        help="use straight through gumbel softmax estimator",
    )
    group.add_argument(
        "--update_gs_temp_frequency",
        default=1,
        type=int,
        help="update gs temperature frequency (default: 1)",
    )
    group.add_argument(
        "--minimum_gs_temperature",
        default=1.0,
        type=float,
        help="minimum gs temperature when frequency update (default: 1.0)",
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module")
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
        help="If set, Sender and Receiver will share the vision encoder",
    )
    group.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision modules will be used",
    )
    group.add_argument("--use_augmentations", action="store_true", default=False)


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=2048,
        help="Projection head's hidden dimension for image features",
    )
    group.add_argument(
        "--projection_output_dim",
        type=int,
        default=2048,
        help="Projection head's output dimension for image features",
    )
    group.add_argument(
        "--simclr_sender",
        default=False,
        action="store_true",
        help="Use a simclr-like sender (no discreteness)",
    )
    group.add_argument(
        "--discrete_evaluation_simclr",
        default=False,
        action="store_true",
        help="Use a simclr-like sender argmaxing the message_like layer at test time",
    )


def get_loss_opts(parser):
    group = parser.add_argument_group("loss")
    group.add_argument(
        "--loss_type",
        type=str,
        default="xent",
        choices=["xent", "ntxent"],
        help="Model name for loss function",
    )
    group.add_argument(
        "--loss_temperature",
        type=float,
        default=0.1,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Similarity function used in loss",
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    parser.add_argument(
        "--use_larc", action="store_true", default=False, help="Use LARC optimizer"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_loss_opts(parser)
    get_game_arch_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
