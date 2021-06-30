# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse

import torch.nn.functional as F
import torch.utils.data

import egg.core as core
from egg.zoo.simple_autoenc.archs import Receiver, Sender
from egg.zoo.simple_autoenc.features import OneHotLoader


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_features",
        type=int,
        default=10,
        help='Dimensionality of the "concept" space (default: 10)',
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=1000,
        help="Number of batches per epoch (default: 1000)",
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="The entropy regularisation coefficient for Sender (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_entropy_coeff",
        type=float,
        default=1e-1,
        help="The entropy regularisation coefficient for Receiver (default: 1e-1)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Sender's parameters (default: 1e-3)",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Receiver's parameters (default: 1e-3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs}"
        "(default: rf)",
    )
    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(
        receiver_output, sender_input.argmax(dim=1), reduction="none"
    )
    return loss, {"acc": acc}


def main(params):
    opts = get_params(params)

    device = torch.device("cuda" if opts.cuda else "cpu")
    train_loader = OneHotLoader(
        n_features=opts.n_features,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
    )
    test_loader = OneHotLoader(
        n_features=opts.n_features,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        seed=7,
    )

    sender = Sender(n_hidden=opts.sender_hidden, n_features=opts.n_features)
    receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)

    if opts.mode.lower() == "rf":
        sender = core.RnnSenderReinforce(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
        )
        callbacks = []
    elif opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )

        receiver = core.RnnReceiverGS(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    optimizer = torch.optim.Adam(
        [
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ]
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks + [core.ConsoleLogger(as_json=True)],
    )
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
