# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse
import contextlib

import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import egg.core as core
from egg.zoo.external_game.archs import Receiver, ReinforceReceiver, Sender
from egg.zoo.external_game.features import CSVDataset


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    parser.add_argument(
        "--dump_data",
        type=str,
        default=None,
        help="Path to the data for which to produce output information",
    )
    parser.add_argument(
        "--dump_output",
        type=str,
        default=None,
        help="Path for dumping output information",
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
        "--sender_layers",
        type=int,
        default=1,
        help="Number of layers in Sender's RNN (default: 1)",
    )
    parser.add_argument(
        "--receiver_layers",
        type=int,
        default=1,
        help="Number of layers in Receiver's RNN (default: 1)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="The entropy regularisation coefficient for Sender (default: 1e-2)",
    )
    parser.add_argument(
        "--receiver_entropy_coeff",
        type=float,
        default=1e-2,
        help="The entropy regularisation coefficient for Receiver (default: 1e-2)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Sender's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Receiver's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gs",
        help="Selects whether GumbelSoftmax or Reinforce is used" "(default: gs)",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=None,
        help="Number of classes for Receiver to output. If not set, is automatically deduced from "
        "the training set",
    )

    args = core.init(parser)
    return args


def dump(game, dataset, device, is_gs):
    interaction = core.dump_interactions(
        game, dataset, gs=is_gs, device=device, variable_length=True
    )

    for i in range(interaction.size):
        sender_input = interaction.sender_input[i]
        message = interaction.message[i]
        receiver_output = interaction.receiver_output[i]
        label = interaction.labels[i]
        length = interaction.message_length[i].long().item()

        sender_input = " ".join(map(str, sender_input.tolist()))
        message = " ".join(map(str, message[:length].tolist()))
        if is_gs:
            receiver_output = receiver_output.argmax()
        print(f"{sender_input};{message};{receiver_output};{label.item()}")


def differentiable_loss(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    labels = labels.squeeze(1)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def non_differentiable_loss(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    labels = labels.squeeze(1)
    acc = (receiver_output == labels).detach().float()
    return -acc, {"acc": acc}


def build_model(opts, train_loader, dump_loader):
    n_features = (
        train_loader.dataset.get_n_features()
        if train_loader
        else dump_loader.dataset.get_n_features()
    )

    if opts.n_classes is not None:
        receiver_outputs = opts.n_classes
    else:
        receiver_outputs = (
            train_loader.dataset.get_output_max() + 1
            if train_loader
            else dump_loader.dataset.get_output_max() + 1
        )

    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)

    if opts.train_mode.lower() == "gs":
        loss = differentiable_loss
        receiver = Receiver(output_size=receiver_outputs, n_hidden=opts.receiver_hidden)
    else:
        loss = non_differentiable_loss
        receiver = ReinforceReceiver(
            output_size=receiver_outputs, n_hidden=opts.receiver_hidden
        )

    return sender, receiver, loss


if __name__ == "__main__":
    opts = get_params()

    print(f"Launching game with parameters: {opts}")

    device = torch.device("cuda" if opts.cuda else "cpu")

    train_loader = None
    if opts.train_data:
        train_loader = DataLoader(
            CSVDataset(path=opts.train_data),
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=1,
        )

    validation_loader = None
    if opts.validation_data:
        validation_loader = DataLoader(
            CSVDataset(path=opts.validation_data),
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=1,
        )

    dump_loader = None
    if opts.dump_data:
        dump_loader = DataLoader(
            CSVDataset(path=opts.dump_data),
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=1,
        )

    assert train_loader or dump_loader, "Either training or dump data must be specified"
    sender, receiver, loss = build_model(opts, train_loader, dump_loader)

    if opts.train_mode.lower() == "rf":
        sender = core.RnnSenderReinforce(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            num_layers=opts.sender_layers,
        )
        receiver = core.RnnReceiverReinforce(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
            num_layers=opts.receiver_layers,
        )

        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            non_differentiable_loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
        )
    elif opts.train_mode.lower() == "gs":
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

        game = core.SenderReceiverRnnGS(sender, receiver, differentiable_loss)
    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
    )

    if dump_loader is not None:
        if opts.dump_output:
            with open(opts.dump_output, "w") as f, contextlib.redirect_stdout(f):
                dump(game, dump_loader, device, opts.train_mode.lower() == "gs")
        else:
            dump(game, dump_loader, device, opts.train_mode.lower() == "gs")
    else:
        trainer.train(n_epochs=opts.n_epochs)

    core.close()
