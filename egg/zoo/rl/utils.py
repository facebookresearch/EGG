# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torchvision.models as models


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def get_opts():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet50"],  # choices=model_names,
        help=f"model architecture: {' | '.join(model_names)} (default: resnet50)",
    )
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--data_workers", type=int, default=2)

    parser.add_argument("--sender_rl", type=float, default=0.001)
    parser.add_argument("--receiver_rl", type=float, default=0.001)

    parser.add_argument("--sender_embedding", type=int, default=128)
    parser.add_argument("--receiver_embedding", type=int, default=128)

    parser.add_argument("--sender_hidden", type=int, default=128)
    parser.add_argument("--receiver_hidden", type=int, default=128)

    parser.add_argument("--vision_hidden_dim_sender", type=int, default=128)
    parser.add_argument("--vision_hidden_dim_receiver", type=int, default=128)
    parser.add_argument("--message_hidden_dim", type=int, default=128)

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

    args = parser.parse_args()
    return args
