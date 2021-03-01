# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import os
import sys


def run_game(game, params):
    os.chdir("/private/home/rdessi/EGG")
    dev_null_file = open(os.devnull, "w")
    old_stdout, sys.stdout = sys.stdout, dev_null_file

    game = importlib.import_module(game)
    params_as_list = [f"--{k}={v}" for k, v in params.items()]
    game.main(params_as_list)

    sys.std_out = old_stdout


def test_continuous_cifar():
    run_game(
        "egg.zoo.simclr.train",
        dict(vocab_size=3, dataset_dir="cifar10", batch_size=32, n_epochs=1),
    )


def test_continuous_imagenet():
    run_game(
        "egg.zoo.simclr.train",
        dict(
            vocab_size=3,
            dataset_dir="/datasets01/imagenet_full_size/061417/val/",
            dataset_name="imagenet",
            batch_size=32,
            n_epochs=1
        ),
    )


def test_rf_cifar():
    run_game(
        "egg.zoo.simclr.train",
        dict(vocab_size=3, dataset_dir="cifar10", batch_size=32, n_epochs=1),
    )


def test_rf_imagenet():
    run_game(
        "egg.zoo.simclr.train",
        dict(
            vocab_size=3,
            dataset_dir="/datasets01/imagenet_full_size/061417/val/",
            dataset_name="imagenet",
            batch_size=32,
            n_epochs=1
        ),
    )
