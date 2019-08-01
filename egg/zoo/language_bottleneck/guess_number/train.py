# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.language_bottleneck.guess_number.features import OneHotLoader, UniformLoader
from egg.zoo.language_bottleneck.guess_number.archs import Sender, Receiver, ReinforcedReceiver

from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from egg.core import EarlyStopperAccuracy


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bits', type=int, default=8,
                        help='')
    parser.add_argument('--bits_s', type=int, default=4,
                        help='')
    parser.add_argument('--bits_r', type=int, default=4,
                        help='')
    parser.add_argument('--n_examples_per_epoch', type=int, default=8000,
                        help='Number of examples seen in an epoch (default: 8000)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Receiver (default: 1e-2)")

    parser.add_argument('--sender_lr', type=float, default=None,
                        help="Learning rate for Sender's parameters (default: 1e-2)")
    parser.add_argument('--receiver_lr', type=float, default=None,
                        help="Learning rate for Receiver's parameters (default: 1e-2)")

    parser.add_argument('--mode', type=str, default='gs',
                        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs,"
                             " non_diff} (default: gs)")

    parser.add_argument('--early_stopping_thr', type=float, default=0.99,
                        help="Early stopping threshold on accuracy (defautl: 0.99)")

    args = core.init(arg_parser=parser, params=params)
    if args.sender_lr is None:
        args.sender_lr = args.lr
    if args.receiver_lr is None:
        args.receiver_lr = args.lr

    assert args.n_examples_per_epoch % args.batch_size == 0
    return args


def diff_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    acc = ((receiver_output > 0.5).long() == labels).detach().all(dim=1).float().mean()
    loss = F.binary_cross_entropy(receiver_output, labels.float(), reduction="none").mean(dim=1)
    return loss, {'acc': acc}


def non_diff_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    acc = ((receiver_output > 0.5).long() == labels).detach().all(dim=1).float()
    return -acc, {'acc': acc.mean()}


def main(params):
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    device = opts.device

    train_loader = OneHotLoader(n_bits=opts.n_bits,
                                bits_s=opts.bits_s,
                                bits_r=opts.bits_r,
                                batch_size=opts.batch_size,
                                batches_per_epoch=opts.n_examples_per_epoch/opts.batch_size)

    test_loader = UniformLoader(n_bits=opts.n_bits, bits_s=opts.bits_s, bits_r=opts.bits_r)
    test_loader.batch = [x.to(device) for x in test_loader.batch]

    sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                    vocab_size=opts.vocab_size)

    if opts.mode == 'gs':
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden, vocab_size=opts.vocab_size)
        sender = core.GumbelSoftmaxWrapper(agent=sender, temperature=opts.temperature)
        game = core.SymbolGameGS(sender, receiver, diff_loss)
    elif opts.mode == 'rf':
        sender = core.ReinforceWrapper(agent=sender)
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden, vocab_size=opts.vocab_size)
        receiver = core.ReinforceDeterministicWrapper(agent=receiver)
        game = core.SymbolGameReinforce(sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff)
    elif opts.mode == 'non_diff':
        sender = core.ReinforceWrapper(agent=sender)
        receiver = ReinforcedReceiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden, vocab_size=opts.vocab_size)
        game = core.SymbolGameReinforce(sender, receiver, non_diff_loss,
                                        sender_entropy_coeff=opts.sender_entropy_coeff,
                                        receiver_entropy_coeff=opts.receiver_entropy_coeff)

    else:
        assert False, 'Unknown training mode'

    optimizer = torch.optim.Adam(
        [
            dict(params=sender.parameters(), lr=opts.sender_lr),
            dict(params=receiver.parameters(), lr=opts.receiver_lr)
        ])

    loss = game.loss

    intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=False,
                                     input_intervention=True)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True), EarlyStopperAccuracy(opts.early_stopping_thr), intervention])

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

