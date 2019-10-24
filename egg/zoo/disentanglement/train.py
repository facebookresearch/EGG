# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import egg.core as core
from egg.zoo.disentanglement.data import ScaledDataset, enumerate_attribute_value, split_train_test
from egg.zoo.disentanglement.archs import Sender, Receiver, PositionalSender, LinearReceiver, BosSender, Shuffler, FactorizedSender, Freezer, Discriminator
from egg.zoo.disentanglement.intervention import Evaluator

from egg.core import EarlyStopperAccuracy


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=4, help='')
    parser.add_argument('--n_values', type=int, default=4, help='')
    parser.add_argument('--data_scaler', type=int, default=100)
    parser.add_argument('--shuffle_messages', action='store_true')
    parser.add_argument('--stats_freq', type=int, default=0)

    parser.add_argument('--sender_hidden', type=int, default=50,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=50,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")

    parser.add_argument('--sender_cell', type=str, default='rnn')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--early_stopping_thr', type=float, default=0.99,
                        help="Early stopping threshold on accuracy (defautl: 0.99)")


    parser.add_argument('--d_weight', type=float, default=0.0,
                        help="")

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        acc = (receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach().float().mean()
        receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
        labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)

        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.n_attributes).mean(dim=-1)
        return loss, {'acc': acc}


def main(params):
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    device = opts.device
    dataset = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    dataset = list(dataset)

    train, test = split_train_test(dataset, 0.1)
    train = ScaledDataset(train, opts.data_scaler)
    test = ScaledDataset(test, 1)

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    test_loader = DataLoader(test, batch_size=opts.batch_size)

    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell == 'linear':
        receiver = LinearReceiver(n_outputs=n_dim, vocab_size=opts.vocab_size, max_length=opts.max_len)
    elif opts.receiver_cell in ['lstm', 'rnn', 'gru']:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverDeterministic(
            receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)
    elif opts.receiver_cell in ['transformer', 'transformer-bos']:
        use_positional_embeddings = opts.receiver_cell == 'transformer-bos' 

        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len, 
                    opts.receiver_hidden, num_heads=1, hidden_size=opts.receiver_hidden, num_layers=1,
                    positional_emb=use_positional_embeddings)
    else:
        raise ValueError(f'Unknown receiver cell, {opts.receiver_cell}')

    if opts.sender_cell == 'positional':
        sender = PositionalSender(n_attributes=opts.n_attributes, n_values=opts.n_values, vocab_size=opts.vocab_size, 
                    max_len=opts.max_len)
    elif opts.sender_cell == 'bos':
        sender = BosSender(n_attributes=opts.n_attributes, n_values=opts.n_values, vocab_size=opts.vocab_size, 
                    max_len=opts.max_len)
    elif opts.sender_cell == 'factorized':
        sender = FactorizedSender(opts.vocab_size, opts.max_len, input_size=opts.n_attributes * opts.n_values, n_hidden=opts.sender_hidden)
    elif opts.sender_cell in ['lstm', 'rnn', 'gru']:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size, 
                embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=True, 
                cell=opts.sender_cell)
    elif opts.sender_cell == 'transformer':
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_hidden, max_len=opts.max_len,
                                                    num_layers=1, num_heads=1, hidden_size=opts.sender_hidden)
    else:
        raise ValueError(f'Unknown sender cell, {opts.sender_cell}')

    if opts.shuffle_messages:
        sender = Shuffler(sender)

    loss = DiffLoss(opts.n_attributes, opts.n_values)

    game = core.SenderReceiverRnnReinforce(
            sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0)


    params = []
    if opts.sender_cell != 'positional':
        params.extend(sender.parameters())

    params.extend(receiver.parameters())
    optimizer = torch.optim.Adam(params, lr=opts.lr)

    evaluator = Evaluator(dataset, opts.device, opts.n_attributes, opts.n_values, opts.vocab_size, freq=opts.stats_freq)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), 
                   EarlyStopperAccuracy(opts.early_stopping_thr, validation=False),
                   evaluator])
    trainer.train(n_epochs=opts.n_epochs)


    # freeze Sender and probe how fast a simple Receiver will learn the thing
    receiver = Receiver(n_hidden=10, n_outputs=n_dim)
    receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_emb, hidden_size=10, cell='lstm')
    sender = Freezer(sender)

    game = core.SenderReceiverRnnReinforce(
            sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0)

    optimizer = torch.optim.Adam(receiver.parameters(), lr=opts.lr)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), 
                   EarlyStopperAccuracy(opts.early_stopping_thr, validation=False)])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
