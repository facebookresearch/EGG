# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_other_opts(parser):
    group = parser.add_argument_group("other")
    group.add_argument("--dummy", type=float, default=1.0, help="dumy option")


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_other_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
