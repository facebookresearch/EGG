# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import random

import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import egg.core as core
from egg.zoo.compositional_efficiency.archs import CircleSender, Lenses, Receiver
from egg.zoo.compositional_efficiency.dataset import SphereData


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver_layers", type=int, default=-1)
    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--cell_layers", type=int, default=1)

    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--receiver_cell", type=str, default="lstm", choices=["lstm", "tree"]
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument("--lenses", type=int, default=0)

    args = core.init(arg_parser=parser, params=params)
    return args


def diff_loss(
    sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input
):
    loss = F.mse_loss(receiver_output, sender_input, reduction="none").mean(-1)
    return loss, {}


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    train_data = SphereData(n_points=int(opts.n_points))
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True)

    test_data = SphereData(n_points=int(1e3))
    test_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=False)

    sender = CircleSender(opts.vocab_size)

    assert opts.lenses in [0, 1]
    if opts.lenses == 1:
        sender = torch.nn.Sequential(Lenses(math.pi / 4), sender)

    receiver = Receiver(
        n_hidden=opts.receiver_hidden, n_dim=2, inner_layers=opts.receiver_layers
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size + 1,  # exclude eos = 0
        opts.receiver_emb,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.cell_layers,
    )

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        diff_loss,
        receiver_entropy_coeff=0.05,
        sender_entropy_coeff=0.0,
    )

    optimizer = core.build_optimizer(receiver.parameters())
    loss = game.loss

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)],
        grad_norm=1.0,
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
