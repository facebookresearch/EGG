# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.summation.features import SequenceLoader
from egg.zoo.summation.archs import Encoder, Receiver


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--max_n', type=int, default=10,
                        help="Max n in a^nb^n(default: 10)")
    args = core.init(parser)

    return args


def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels)
    return loss, {'acc': acc}


if __name__ == "__main__":
    opts = get_params()

    device = torch.device("cuda" if opts.cuda else "cpu")

    train_loader = SequenceLoader(max_n=opts.max_n, batch_size=opts.batch_size,
                                  batches_per_epoch=opts.batches_per_epoch)
    test_loader = SequenceLoader(max_n=opts.max_n, batch_size=opts.batch_size,
                                 batches_per_epoch=opts.batches_per_epoch, seed=7)

    encoder = Encoder(n_hidden=opts.sender_output_size,
                      emb_dim=opts.sender_embedding_size,
                      cell=opts.sender_cell, vocab_size=3)  # only 3 symbols in the incoming data
    sender = core.build_sender(encoder, opts)

    receiver = Receiver(opts.receiver_input_size)
    receiver = core.build_receiver(receiver, opts, deterministic=True)

    game = core.build_game(sender, receiver, loss, opts)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader)
    trainer.train(n_epochs=opts.n_epochs)

    sender_inputs, messages, _, receiver_outputs, labels = \
        core.dump_sender_receiver(game, test_loader, gs=opts.mode == 'gs', device=device, variable_length=opts.variable_length)

    for (seq, l), message, output, label in zip(sender_inputs, messages, receiver_outputs, labels):
        print(f'{seq[:l]} -> {message} -> {output.argmax()} (label = {label})')

    core.close()

