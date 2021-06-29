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
from egg.zoo.compositional_efficiency.archs import (
    IdentitySender,
    Receiver,
    RotatedSender,
)
from egg.zoo.compositional_efficiency.dataset import AttributeValueData


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver_layers", type=int, default=-1)
    parser.add_argument("--cell_layers", type=int, default=1)

    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument("--receiver_cell", type=str, default="rnn")
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )

    parser.add_argument("--n_a", type=int, default=2)
    parser.add_argument("--n_v", type=int, default=10)

    parser.add_argument("--language", type=str, choices=["identity", "rotated"])

    parser.add_argument(
        "--loss_type", choices=["autoenc", "mixed", "linear"], default="autoenc"
    )

    args = core.init(arg_parser=parser, params=params)
    return args


SMALL_PRIMES = [3, 5, 7, 11, 13, 17]


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, loss_type):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

        self.loss_type = loss_type
        self.coeffs = SMALL_PRIMES

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.loss_type == "mixed":
            left = (sender_input[:, 0] + sender_input[:, 1]).fmod(self.n_values)
            right = (sender_input[:, 0] - sender_input[:, 1] + self.n_values).fmod(
                self.n_values
            )
            sender_input[:, 0], sender_input[:, 1] = left, right
        elif self.loss_type == "linear":
            a, b, c, d, e, f = self.coeffs
            left = (a * sender_input[:, 0] + b * sender_input[:, 1] + c).fmod(
                self.n_values
            )
            right = (d * sender_input[:, 0] + e * sender_input[:, 1] + f).fmod(
                self.n_values
            )
            sender_input[:, 0], sender_input[:, 1] = left, right
        elif self.loss_type == "autoenc":
            pass
        else:
            assert False

        acc = (
            torch.sum((receiver_output.argmax(dim=-1) == sender_input).detach(), dim=1)
            == self.n_attributes
        ).float()
        acc_or = (receiver_output.argmax(dim=-1) == sender_input).float()

        receiver_output = receiver_output.view(
            batch_size * self.n_attributes, self.n_values
        )
        labels = sender_input.view(batch_size * self.n_attributes)

        loss = (
            F.cross_entropy(receiver_output, labels, reduction="none")
            .view(batch_size, self.n_attributes)
            .mean(dim=-1)
        )

        return loss, {"acc": acc, "acc_or": acc_or}


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    n_a, n_v = opts.n_a, opts.n_v
    opts.vocab_size = n_v

    train_data = AttributeValueData(n_attributes=n_a, n_values=n_v, mul=1, mode="train")
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True)

    test_data = AttributeValueData(n_attributes=n_a, n_values=n_v, mul=1, mode="test")
    test_loader = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False)

    print(f"# Size of train {len(train_data)} test {len(test_data)}")

    if opts.language == "identity":
        sender = IdentitySender(n_a, n_v)
    elif opts.language == "rotated":
        sender = RotatedSender(n_a, n_v)
    else:
        assert False

    receiver = Receiver(
        n_hidden=opts.receiver_hidden,
        n_dim=n_a * n_v,
        inner_layers=opts.receiver_layers,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size + 1,  # exclude eos = 0
        opts.receiver_emb,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.cell_layers,
    )

    diff_loss = DiffLoss(n_a, n_v, loss_type=opts.loss_type)

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        diff_loss,
        receiver_entropy_coeff=0.05,
        sender_entropy_coeff=0.0,
    )

    optimizer = core.build_optimizer(receiver.parameters())
    loss = game.loss

    early_stopper = core.EarlyStopperAccuracy(1.0, validation=False)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            early_stopper,
        ],
        grad_norm=1.0,
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
