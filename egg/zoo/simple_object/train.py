# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch.nn.functional as F

from egg.core.datasets import AttributesValuesDataset, AttributesValuesWithDistractorsDataset
import egg.core as core
from egg.zoo.simple_object.archs import Sender, Receiver


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=3,
                        help='Dimensionality of the "concept" space (default: 3)')
    parser.add_argument('--n_values', type=int, default=30,
                        help='Dimensionality of each "concept" (default: 10)')

    parser.add_argument('--distractors', type=int, default=0,
                        help='Number of distractors" (default: 0)')

    parser.add_argument('--samples_per_epoch', type=int, default=5000,
                        help='Number of samples per epoch (default: 5000)')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples (default: 1000)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')

    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    args = core.init(parser, params)

    return args


def reconstruction_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


def discrimination_loss(_sender_input,  _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {'acc': acc}


def main(params):
    opts = get_params(params)

    if opts.distractors:
        train_loader = AttributesValuesWithDistractorsDataset(n_attributes=opts.n_attributes,
                                               n_values=opts.n_values,
                                               n_train_samples_per_epoch=opts.samples_per_epoch,
                                               batch_size=opts.batch_size,
                                               distractors=opts.distractors,
                                               seed=opts.random_seed)
        test_loader = AttributesValuesWithDistractorsDataset(n_attributes=opts.n_attributes,
                                               n_values=opts.n_values,
                                               n_train_samples_per_epoch=opts.val_samples,
                                               batch_size=opts.batch_size,
                                               distractors=opts.distractors,
                                               seed=opts.random_seed+1)
    else:
        train_loader = AttributesValuesDataset(n_attributes=opts.n_attributes,
                                               n_values=opts.n_values,
                                               n_train_samples_per_epoch=opts.samples_per_epoch,
                                               batch_size=opts.batch_size,
                                               seed=opts.random_seed)
        test_loader = AttributesValuesDataset(n_attributes=opts.n_attributes,
                                               n_values=opts.n_values,
                                               n_train_samples_per_epoch=opts.val_samples,
                                               batch_size=opts.batch_size,
                                               seed=opts.random_seed+1)

    sender = Sender(n_hidden=opts.sender_hidden, n_features=opts.n_attributes)
    receiver = Receiver(n_features=opts.n_attributes, n_hidden=opts.receiver_hidden)

    sender = core.RnnSenderReinforce(sender, opts.vocab_size, opts.sender_embedding, opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len)

    receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding, opts.receiver_hidden, cell=opts.receiver_cell)

    loss = discrimination_loss if opts.distractors else reconstruction_loss

    game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=opts.receiver_entropy_coeff)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(print_train_loss=True, as_json=True)])

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
