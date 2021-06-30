# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse

import torch.nn.functional as F
import torch.utils.data

import egg.core as core
from egg.zoo.summation.archs import Encoder, Receiver
from egg.zoo.summation.features import SequenceLoader


def get_params():
    parser = argparse.ArgumentParser()
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
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--max_n", type=int, default=10, help="Max n in a^nb^n(default: 10)"
    )
    args = core.init(parser)

    return args


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels)
    return loss, {"acc": acc}


if __name__ == "__main__":
    opts = get_params()

    device = torch.device("cuda" if opts.cuda else "cpu")

    train_loader = SequenceLoader(
        max_n=opts.max_n,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
    )
    test_loader = SequenceLoader(
        max_n=opts.max_n,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        seed=7,
    )

    encoder = Encoder(
        n_hidden=opts.sender_hidden,
        embed_dim=opts.sender_embedding,
        cell=opts.sender_cell,
        vocab_size=3,
    )  # only 3 symbols in the incoming data
    sender = core.RnnSenderGS(
        encoder,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        temperature=opts.temperature,
    )

    receiver = Receiver(opts.receiver_hidden)
    receiver = core.RnnReceiverGS(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
    )

    game = core.SenderReceiverRnnGS(sender, receiver, loss)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
    )
    trainer.train(n_epochs=opts.n_epochs)

    sender_inputs, messages, _, receiver_outputs, labels = core.dump_interactions(
        game, test_loader, gs=True, device=device, variable_length=True
    )

    for (seq, l), message, output, label in zip(
        sender_inputs, messages, receiver_outputs, labels
    ):
        print(f"{seq[:l]} -> {message} -> {output.argmax()} (label = {label})")

    core.close()
