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

def dump(game, dataset, device, is_gs, is_var_length):
    sender_inputs, messages, _1, receiver_outputs, _2 = \
        core.dump_sender_receiver(game, dataset, gs=is_gs, device=device, variable_length=is_var_length)

    for sender_input, message, receiver_output \
            in zip(sender_inputs, messages, receiver_outputs):
        sender_input = ''.join(map(str, sender_input.tolist()))
        if is_var_length:
            message = ' '.join(map(str, message.tolist()))
        receiver_output = (receiver_output > 0.5).tolist()
        receiver_output = ''.join([str(x) for x in receiver_output])
        print(f'{sender_input} -> {message} -> {receiver_output}')


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
                        help="Learning rate for Sender's parameters")
    parser.add_argument('--receiver_lr', type=float, default=None,
                        help="Learning rate for Receiver's parameters")

    parser.add_argument('--mode', type=str, default='gs',
                        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs,"
                             " non_diff} (default: gs)")

    parser.add_argument('--variable_length', action='store_true', default=False)
    parser.add_argument('--sender_cell', type=str, default='rnn')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--early_stopping_thr', type=float, default=0.99,
                        help="Early stopping threshold on accuracy (defautl: 0.99)")
    parser.add_argument('--dump_language', action='store_true')

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


    if not opts.variable_length:
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.vocab_size)
        if opts.mode == 'gs':
            sender = core.GumbelSoftmaxWrapper(agent=sender, temperature=opts.temperature)
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameGS(sender, receiver, diff_loss)
        elif opts.mode == 'rf':
            sender = core.ReinforceWrapper(agent=sender)
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            receiver = core.ReinforceDeterministicWrapper(agent=receiver)
            game = core.SymbolGameReinforce(sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff)
        elif opts.mode == 'non_diff':
            sender = core.ReinforceWrapper(agent=sender)
            receiver = ReinforcedReceiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden, vocab_size=opts.vocab_size)
            game = core.SymbolGameReinforce(sender, receiver, non_diff_loss,
                                            sender_entropy_coeff=opts.sender_entropy_coeff,
                                            receiver_entropy_coeff=opts.receiver_entropy_coeff)
    else:
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                vocab_size=opts.sender_hidden) # TODO: not really vocab
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        if opts.mode == 'gs':
            sender = core.RnnSenderGS(agent=sender, vocab_size=opts.vocab_size, temperature=opts.temperature,
                    emb_dim=opts.sender_emb, n_hidden=opts.sender_hidden, max_len=opts.max_len, force_eos=True, cell=opts.sender_cell)

            receiver = core.RnnReceiverGS(receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)

            game = core.SenderReceiverRnnGS(sender, receiver, diff_loss)
        elif opts.mode == 'rf':
            sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size, 
                    emb_dim=opts.sender_emb, n_hidden=opts.sender_hidden, max_len=opts.max_len, force_eos=True, cell=opts.sender_cell)
            receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)
            game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=opts.receiver_entropy_coeff)
        else:
            assert False

    optimizer = torch.optim.Adam(
        [
            dict(params=sender.parameters(), lr=opts.sender_lr),
            dict(params=receiver.parameters(), lr=opts.receiver_lr)
        ])

    loss = game.loss

    intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
                                     input_intervention=True)

<<<<<<< HEAD
    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True), EarlyStopperAccuracy(opts.early_stopping_thr), intervention])
=======
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr)

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader, epoch_callback=intervention,
                           early_stopping=early_stopper, callbacks=[core.ConsoleLogger(as_json=True)])
>>>>>>> wip

    trainer.train(n_epochs=opts.n_epochs)


    if opts.dump_language:
        dump(game, test_loader, device, is_gs=opts.mode == 'gs', is_var_length=opts.variable_length)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

