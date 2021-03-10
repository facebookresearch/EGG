# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


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


def get_opts(params):
    parser = argparse.ArgumentParser()

    # Data opts
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/datasets01/imagenet_full_size/061417/train",
        help="Dataset location",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers used in the dataloader"
    )

    # Vision module opts
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
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
        "--receiver_output_size",
        type=int,
        default=128,
        help="Receiver output size"
    )

    # Misc opts
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    parser.add_argument(
        "--is_distributed",
        action="store_true",
        default=False
    )
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
