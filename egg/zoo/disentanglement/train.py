# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import egg.core as core
from egg.zoo.disentanglement.data import ScaledDataset, enumerate_attribute_value, split_train_test, one_hotify, split_holdout, select_subset
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
    parser.add_argument('--density_data', type=int, default=0, help='no sampling if equal 0')

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
    parser.add_argument('--early_stopping_thr', type=float, default=0.99999,
                        help="Early stopping threshold on accuracy (defautl: 0.99999)")

    parser.add_argument('--d_type', default=None, choices=['pos', 'bos', None])

    parser.add_argument('--d_weight', type=float, default=0.0, help="")

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, \
                  generalization=False):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        if self.test_generalization:
            acc, acc_or, loss = 0.0, 0.0, 0.0

            for attr in range(self.n_attributes):
                zero_index = sender_input[:, attr, 0].nonzero().squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat([masked_input[:, :attr, :], masked_input[:, attr+1:, :]], dim=1)
                no_attribute_output = torch.cat([masked_output[:, :attr, :], masked_output[:, attr+1:, :]], dim=1)

                n_attributes = self.n_attributes - 1
                attr_acc = ((no_attribute_output.argmax(dim=-1) == no_attribute_input.argmax(dim=-1)).sum(dim=1) == n_attributes).float().mean()
                acc += attr_acc

                attr_acc_or = (no_attribute_output.argmax(dim=-1) == no_attribute_input.argmax(dim=-1)).float().mean()
                acc_or += attr_acc_or


                #receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
                labels = no_attribute_input.argmax(dim=-1).view(masked_size * n_attributes)
                predictions = no_attribute_output.view(masked_size * n_attributes, self.n_values)
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc /= self.n_attributes
            acc_or /= self.n_attributes
        else:
            acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach(), dim=1) == self.n_attributes).float().mean()
            acc_or = (receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).float().mean()

            receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
            labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
            loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.n_attributes).mean(dim=-1)

        return loss, {'acc': acc, 'acc_or': acc_or}


def _set_seed(seed) -> None:
    import random
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(params):
    import copy
    opts = get_params(params)
    to_print = copy.deepcopy(vars(opts))
    del to_print['device']
    print(json.dumps(to_print))

    device = opts.device
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    if opts.density_data>0:
        sampled_data = select_subset(full_data, opts.density_data, opts.n_attributes, opts.n_values)
        full_data = copy.deepcopy(sampled_data)

    train, generalization_holdout = split_holdout(full_data)
    train, uniform_holdout = split_train_test(train, 0.1)

    generalization_holdout, train, uniform_holdout, full_data = [one_hotify(x, opts.n_attributes, opts.n_values) for x in [generalization_holdout, train, uniform_holdout, full_data]]

    train, validation = ScaledDataset(train, opts.data_scaler), ScaledDataset(train, 1)

    generalization_holdout, uniform_holdout, full_data = ScaledDataset(generalization_holdout), ScaledDataset(uniform_holdout), ScaledDataset(full_data)
    generalization_holdout_loader, uniform_holdout_loader, full_data_loader = [DataLoader(x, batch_size=opts.batch_size) for x in [generalization_holdout, uniform_holdout, full_data]]

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=validation.__len__())

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

    if discriminator:
        game = SenderReceiverRnnReinforceWithDiscriminator(
                sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0, discriminator=discriminator, discriminator_weight=opts.d_weight,
                discriminator_transform=discriminator_transfo)
    else:
        game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                                       receiver_entropy_coeff=0.0, length_cost=0.0)
    optimizer = torch.optim.Adam(params, lr=opts.lr)

    metrics_evaluator = Metrics(validation.examples, opts.device, opts.n_attributes, opts.n_values, opts.vocab_size + 1, freq=opts.stats_freq)

    loaders = []
    loaders.append(("generalization hold out", generalization_holdout_loader, DiffLoss(opts.n_attributes, opts.n_values, generalization=True)))
    loaders.append(("uniform holdout", uniform_holdout_loader,  DiffLoss(opts.n_attributes, opts.n_values)))

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=False),
                   early_stopper,
                   metrics_evaluator,
                   holdout_evaluator])
    trainer.train(n_epochs=opts.n_epochs)
    validation_acc = early_stopper.validation_stats[-1][1]['acc']

    #dump(game, full_data_loader, opts.device, opts.n_attributes, opts.n_values)

    # Train new agents
    if validation_acc > 0.99:
        core.get_opts().preemptable = False
        core.get_opts().checkpoint_path = None

        # freeze Sender and probe how fast a simple Receiver will learn the thing
        def retrain_receiver(receiver_generator, sender):
            receiver = receiver_generator()
            game = SenderReceiverRnnReinforceWithDiscriminator(
                    sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0)
            optimizer = torch.optim.Adam(receiver.parameters(), lr=opts.lr)
            early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)

            trainer = core.Trainer(
                game=game, optimizer=optimizer,
                train_data=train_loader,
                validation_data=validation_loader,
                callbacks=[
                        early_stopper,
                        Evaluator(loaders, opts.device, freq=0)
                        ])
            trainer.train(n_epochs=opts.n_epochs)

            accs = [x[1]['acc'] for x in early_stopper.validation_stats]
            return accs

        # freeze Sender and probe how fast a simple Receiver will learn the thing
        def retrain_sender(speaker_generator, receiver):
            sender = sender_generator()
            game = SenderReceiverRnnReinforceWithDiscriminator(
                    sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0)
            optimizer = torch.optim.Adam(sender.parameters(), lr=opts.lr)
            early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)
            trainer = core.Trainer(
                game=game, optimizer=optimizer,
                train_data=train_loader,
                validation_data=validation_loader,
                callbacks=[
                        early_stopper,
                        Evaluator(loaders, opts.device, freq=0) # Only evaluate on the uniform heldout
                        ])
            trainer.train(n_epochs=opts.n_epochs)

            accs = [x[1]['acc'] for x in early_stopper.validation_stats]
            return accs

        frozen_sender = Freezer(copy.deepcopy(sender))
        frozen_receiver = Freezer(copy.deepcopy(receiver))

        gru_receiver_generator = lambda: \
            core.RnnReceiverDeterministic(Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim),
                    opts.vocab_size + 1, opts.receiver_emb, hidden_size=opts.receiver_hidden, cell='gru')

        transformer_receiver_generator = lambda: \
            core.TransformerReceiverDeterministic(
                    Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim),
                    opts.vocab_size + 1, opts.max_len,
                    opts.receiver_hidden, num_heads=5, hidden_size=opts.receiver_emb, num_layers=1,
                    positional_emb=True)

        linear_receiver_generator = lambda: \
            core.RnnReceiverDeterministic(Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim),
                    opts.vocab_size + 1, opts.receiver_emb, hidden_size=opts.receiver_hidden, cell='gru')

        for name, receiver_generator in [('transformer', transformer_receiver_generator), ('gru', gru_receiver_generator), ('linear', linear_receiver_generator)]:
            for seed in range(17, 17 + 3):
                _set_seed(seed)
                accs = retrain_receiver(receiver_generator, frozen_sender)
                accs += [1.0] * (opts.n_epochs - len(accs))
                auc = sum(accs)
                print(json.dumps({"mode": "reset", "seed": seed, "receiver_name": name, "auc": auc}))

        gru_sender_generator = lambda: \
            PlusOneWrapper(core.RnnSenderReinforce(agent=Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden),
                    vocab_size=opts.vocab_size, embed_dim=opts.sender_emb,
                    hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=False,
                    cell='gru'))

        transformer_sender_generator = lambda: \
            PlusOneWrapper(core.TransformerSenderReinforce(agent=Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden),
                    vocab_size=opts.vocab_size, embed_dim=opts.sender_hidden, max_len=opts.max_len,
                    num_layers=1, num_heads=5, hidden_size=opts.sender_emb, force_eos=False))

        for name, sender_generator in [('transformer', transformer_sender_generator), ('gru', gru_sender_generator)]:
            for seed in range(17, 17 + 3):
                _set_seed(seed)
                accs = retrain_sender(sender_generator, frozen_receiver)
                accs += [1.0] * (opts.n_epochs - len(accs))
                aucs = sum(accs)
                print(json.dumps({"mode": "reset", "seed": seed, "sender_name": name, "auc": auc}))

    core.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
