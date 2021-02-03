# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torchvision.models as models

import egg.core as core


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def get_opts(params):
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet50"],  # choices=model_names,
        help=f"model architecture: {' | '.join(model_names)} (default: resnet50)",
    )

    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument("--max_targets_seen", type=int, default=100)

    parser.add_argument("--sender_lr", type=float, default=0.001)
    parser.add_argument("--receiver_lr", type=float, default=0.001)

    parser.add_argument("--sender_embedding", type=int, default=128)
    parser.add_argument("--receiver_embedding", type=int, default=128)

    parser.add_argument("--sender_hidden", type=int, default=128)
    parser.add_argument("--receiver_hidden", type=int, default=128)
    parser.add_argument("--vision_hidden_dim_receiver", type=int, default=128)

    parser.add_argument("--similarity_projection", type=int, default=128)

    parser.add_argument(
        "--sender_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )

    parser.add_argument(
        "--receiver_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )

    parser.add_argument("--pdb", action="store_true", default=False)

    args = core.init(arg_parser=parser, params=params)
    return args
