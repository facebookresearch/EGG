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
        default="imagenet",
        choices=["cifar10", "imagenet"],
        help="Dataset name",
    )
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="/datasets01/imagenet_full_size/061417/train",
        help="Dataset location",
    )
    group.add_argument(
        "--validation_dataset_dir",
        type=str,
        default="",
        help="Dataset location",
    )
    group.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size"
    )
    group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers used in the dataloader"
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=0.3,
        help="gs temperature used in the relaxation layer"
    )
    group.add_argument(
        "--train_gs_temperature",
        default=False,
        action="store_true",
        help="train gs temperature used in the relaxation layer"
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
    group = parser.add_argument_group("game architecture")
    group.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=2048,
        help="Projection head's hidden dimension for image features."
    )
    group.add_argument(
        "--projection_output_dim",
        type=int,
        default=2048,
        help="Projection head's output dimension for image features."
    )


def get_loss_opts(parser):
    group = parser.add_argument_group("loss")
    group.add_argument(
        "--loss_temperature",
        type=float,
        default=0.1,
        help="Temperature for (NT)XEnt loss",
    )
    group.add_argument(
        "--use_ntxent",
        default=True,
        action="store_true",
        help="Temperature for (NT)XEnt loss",
    )
    group.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        help="Similarity used in  (NT)XEnt loss",
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
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
