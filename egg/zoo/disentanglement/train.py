# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import egg.core as core
from egg.zoo.disentanglement.data import ScaledDataset, enumerate_attribute_value, split_train_test, one_hotify, split_by_attribute_value
from egg.zoo.disentanglement.archs import Sender, Receiver, PositionalSender, LinearReceiver, BosSender, Shuffler, FactorizedSender, \
    Freezer, PositionalDiscriminator, SenderReceiverRnnReinforceWithDiscriminator, PlusOneWrapper, HistogramDiscriminator
from egg.zoo.disentanglement.intervention import Metrics, Evaluator, histogram

from egg.core import EarlyStopperAccuracy

def dump(game, dataset, device, n_attributes, n_values):
    sender_inputs, messages, _1, receiver_outputs, _2 = \
        core.dump_sender_receiver(
            game, dataset, gs=False, device=device, variable_length=True)

    language = []

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        sender_input = sender_input.view(n_attributes, n_values).argmax(dim=-1).view(-1).tolist()
        sender_input = " ".join(str(x) for x in sender_input)
        message = " ".join(str(x) for x in message.tolist())
        receiver_output = receiver_output.view(n_attributes, n_values).argmax(dim=-1).view(-1).tolist()
        receiver_output = " ".join(str(x) for x in receiver_output)

        utterance = dict(sender_input=sender_input, message=message, receiver_output=receiver_output)
        language.append(utterance)

    language = json.dumps({'language': language})
    print(language)


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


    parser.add_argument('--d_type', default=None, choices=['pos', 'bos'])

    parser.add_argument('--d_weight', type=float, default=0.0, help="")

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, cut_at_attribute=None):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.cut_attribute = cut_at_attribute

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        if self.cut_attribute is not None:
            sender_input = torch.cat([sender_input[:, :self.cut_attribute, :], sender_input[:, self.cut_attribute+1:, :]], dim=1) 
            receiver_output = torch.cat([receiver_output[:, :self.cut_attribute, :],receiver_output[:, self.cut_attribute+1:, :]], dim=1)
            n_attributes = self.n_attributes - 1
        else:
            n_attributes = self.n_attributes

        acc = (receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach().float().mean()
        receiver_output = receiver_output.view(batch_size * n_attributes, self.n_values)
        labels = sender_input.argmax(dim=-1).view(batch_size * n_attributes)

        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, n_attributes).mean(dim=-1)
        return loss, {'acc': acc}


def _set_seed(seed) -> None:
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(params):
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    device = opts.device
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)

    holdout_a1, rest = split_by_attribute_value(full_data, 0, 0)
    holdout_a2, rest = split_by_attribute_value(rest, 1, 0)
    train, uniform_holdout = split_train_test(rest, 0.1)

    apply = lambda x: list(one_hotify(x, opts.n_attributes, opts.n_values))
    holdout_a1, holdout_a2, train, uniform_holdout, full_data = list(map(apply, [holdout_a1, holdout_a2, train, uniform_holdout, full_data]))

    train, validation = ScaledDataset(train, opts.data_scaler), ScaledDataset(train, 1)

    holdout_a1, holdout_a2, uniform_holdout, full_data = ScaledDataset(holdout_a1), ScaledDataset(holdout_a2), ScaledDataset(uniform_holdout), ScaledDataset(full_data)
    holdout_a1_loader, holdout_a2_loader, uniform_holdout_loader, full_data_loader = [DataLoader(x, batch_size=opts.batch_size) for x in [holdout_a1, holdout_a2, uniform_holdout, full_data]]

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=opts.batch_size)

    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell == 'linear':
        receiver = LinearReceiver(n_outputs=n_dim, vocab_size=opts.vocab_size + 1, max_length=opts.max_len)
    elif opts.receiver_cell in ['lstm', 'rnn', 'gru']:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverDeterministic(
            receiver, opts.vocab_size + 1, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)
    elif opts.receiver_cell in ['transformer', 'transformer-bos']:
        use_positional_embeddings = opts.receiver_cell != 'transformer-bos' 

        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size + 1, opts.max_len, 
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
                embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=False, 
                cell=opts.sender_cell)
    elif opts.sender_cell == 'transformer':
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_hidden, max_len=opts.max_len,
                                                    num_layers=1, num_heads=1, hidden_size=opts.sender_hidden, force_eos=False)
    else:
        raise ValueError(f'Unknown sender cell, {opts.sender_cell}')

    if opts.shuffle_messages:
        sender = Shuffler(sender)

    sender = PlusOneWrapper(sender)
    loss = DiffLoss(opts.n_attributes, opts.n_values)

    params = []
    if opts.sender_cell != 'positional':
        params.extend(sender.parameters())
    params.extend(receiver.parameters())

    if opts.d_type == 'pos':
        discriminator = PositionalDiscriminator(opts.vocab_size + 1, n_hidden=opts.receiver_hidden, embed_dim=opts.receiver_emb)
        discriminator_transfo = None
        params.extend(discriminator.parameters())
    elif opts.d_type == 'bos':
        discriminator = HistogramDiscriminator(opts.vocab_size, opts.receiver_hidden, opts.receiver_emb)
        discriminator_transfo = lambda x: histogram(x, opts.vocab_size)
        params.extend(discriminator.parameters())
    else:
        discriminator = None
        discriminator_transfo = None


    game = SenderReceiverRnnReinforceWithDiscriminator(
            sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0, discriminator=discriminator, discriminator_weight=opts.d_weight,
            discriminator_transform=discriminator_transfo)


    optimizer = torch.optim.Adam(params, lr=opts.lr)

    metrics_evaluator = Metrics(full_data.examples, opts.device, opts.n_attributes, opts.n_values, opts.vocab_size + 1, freq=opts.stats_freq)

    loaders = []
    loaders.append(("hold out a1", holdout_a1_loader, DiffLoss(opts.n_attributes, opts.n_values, 0)))
    loaders.append(("hold out a2", holdout_a2_loader, DiffLoss(opts.n_attributes, opts.n_values, 1)))
    loaders.append(("uniform holdout", uniform_holdout_loader,  DiffLoss(opts.n_attributes, opts.n_values)))

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), 
                   EarlyStopperAccuracy(opts.early_stopping_thr, validation=True),
                   metrics_evaluator,
                   holdout_evaluator])
    trainer.train(n_epochs=opts.n_epochs)

    dump(game, full_data_loader, opts.device, opts.n_attributes, opts.n_values)

    # freeze Sender and probe how fast a simple Receiver will learn the thing
    sender = Freezer(sender)
    core.get_opts().preemptable = False
    core.get_opts().checkpoint_path = None

    def retrain_receiver(receiver_generator):
        receiver = receiver_generator()
        game = SenderReceiverRnnReinforceWithDiscriminator(
                sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0)
        optimizer = torch.optim.Adam(receiver.parameters(), lr=opts.lr)

        trainer = core.Trainer(
            game=game, optimizer=optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=[#core.ConsoleLogger(as_json=True, print_train_loss=False),
                    EarlyStopperAccuracy(opts.early_stopping_thr, validation=True),
                    Evaluator(loaders, opts.device, freq=0)
                    ])
        trainer.train(n_epochs=opts.n_epochs)

    lstm_receiver_generator = lambda: \
        core.RnnReceiverDeterministic(Receiver(n_hidden=10, n_outputs=n_dim),
                opts.vocab_size + 1, opts.receiver_emb, hidden_size=10, cell='lstm')

    transformer_receiver_generator = lambda: \
        core.TransformerReceiverDeterministic(
                Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim),
                opts.vocab_size + 1, opts.max_len, 
                opts.receiver_hidden, num_heads=1, hidden_size=opts.receiver_hidden, num_layers=1,
                positional_emb=True)

    for name, receiver_generator in [('transformer', transformer_receiver_generator), ('lstm', lstm_receiver_generator)]:
        for seed in range(17, 17 + 3):
            _set_seed(seed)
            print(json.dumps({"mode": "reset", "seed": seed, "receiver_name": name}))
            retrain_receiver(receiver_generator)

    core.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
