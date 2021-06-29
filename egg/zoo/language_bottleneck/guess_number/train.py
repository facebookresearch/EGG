# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch.nn.functional as F
import torch.utils.data

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.language_bottleneck.guess_number.archs import (
    Receiver,
    ReinforcedReceiver,
    Sender,
)
from egg.zoo.language_bottleneck.guess_number.features import (
    OneHotLoader,
    UniformLoader,
)
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bits", type=int, default=8, help="")
    parser.add_argument("--bits_s", type=int, default=4, help="")
    parser.add_argument("--bits_r", type=int, default=4, help="")
    parser.add_argument(
        "--n_examples_per_epoch",
        type=int,
        default=8000,
        help="Number of examples seen in an epoch (default: 8000)",
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
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )
    parser.add_argument(
        "--receiver_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Receiver (default: 1e-2)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=None,
        help="Learning rate for Sender's parameters",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=None,
        help="Learning rate for Receiver's parameters",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs,"
        " non_diff} (default: gs)",
    )

    parser.add_argument("--variable_length", action="store_true", default=False)
    parser.add_argument("--sender_cell", type=str, default="rnn")
    parser.add_argument("--receiver_cell", type=str, default="rnn")
    parser.add_argument(
        "--sender_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.99,
        help="Early stopping threshold on accuracy (defautl: 0.99)",
    )

    args = core.init(arg_parser=parser, params=params)
    if args.sender_lr is None:
        args.sender_lr = args.lr
    if args.receiver_lr is None:
        args.receiver_lr = args.lr

    assert args.n_examples_per_epoch % args.batch_size == 0
    return args


def diff_loss(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    acc = ((receiver_output > 0.5).long() == labels).detach().all(dim=1).float()
    loss = F.binary_cross_entropy(
        receiver_output, labels.float(), reduction="none"
    ).mean(dim=1)
    return loss, {"acc": acc}


def non_diff_loss(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    acc = ((receiver_output > 0.5).long() == labels).detach().all(dim=1).float()
    return -acc, {"acc": acc.mean()}


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    train_loader = OneHotLoader(
        n_bits=opts.n_bits,
        bits_s=opts.bits_s,
        bits_r=opts.bits_r,
        batch_size=opts.batch_size,
        batches_per_epoch=opts.n_examples_per_epoch / opts.batch_size,
    )

    test_loader = UniformLoader(
        n_bits=opts.n_bits, bits_s=opts.bits_s, bits_r=opts.bits_r
    )
    test_loader.batch = [x.to(device) for x in test_loader.batch]

    if not opts.variable_length:
        sender = Sender(
            n_bits=opts.n_bits, n_hidden=opts.sender_hidden, vocab_size=opts.vocab_size
        )
        if opts.mode == "gs":
            sender = core.GumbelSoftmaxWrapper(
                agent=sender, temperature=opts.temperature
            )
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            receiver = core.SymbolReceiverWrapper(
                receiver,
                vocab_size=opts.vocab_size,
                agent_input_size=opts.receiver_hidden,
            )
            game = core.SymbolGameGS(sender, receiver, diff_loss)
        elif opts.mode == "rf":
            sender = core.ReinforceWrapper(agent=sender)
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            receiver = core.SymbolReceiverWrapper(
                receiver,
                vocab_size=opts.vocab_size,
                agent_input_size=opts.receiver_hidden,
            )
            receiver = core.ReinforceDeterministicWrapper(agent=receiver)
            game = core.SymbolGameReinforce(
                sender,
                receiver,
                diff_loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
            )
        elif opts.mode == "non_diff":
            sender = core.ReinforceWrapper(agent=sender)
            receiver = ReinforcedReceiver(
                n_bits=opts.n_bits, n_hidden=opts.receiver_hidden
            )
            receiver = core.SymbolReceiverWrapper(
                receiver,
                vocab_size=opts.vocab_size,
                agent_input_size=opts.receiver_hidden,
            )

            game = core.SymbolGameReinforce(
                sender,
                receiver,
                non_diff_loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
                receiver_entropy_coeff=opts.receiver_entropy_coeff,
            )
    else:
        if opts.mode != "rf":
            print("Only mode=rf is supported atm")
            opts.mode = "rf"

        if opts.sender_cell == "transformer":
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            sender = Sender(
                n_bits=opts.n_bits,
                n_hidden=opts.sender_hidden,
                vocab_size=opts.sender_hidden,
            )  # TODO: not really vocab
            sender = core.TransformerSenderReinforce(
                agent=sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_emb,
                max_len=opts.max_len,
                num_layers=1,
                num_heads=1,
                hidden_size=opts.sender_hidden,
            )
        else:
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            sender = Sender(
                n_bits=opts.n_bits,
                n_hidden=opts.sender_hidden,
                vocab_size=opts.sender_hidden,
            )  # TODO: not really vocab
            sender = core.RnnSenderReinforce(
                agent=sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_emb,
                hidden_size=opts.sender_hidden,
                max_len=opts.max_len,
                cell=opts.sender_cell,
            )

        if opts.receiver_cell == "transformer":
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_emb)
            receiver = core.TransformerReceiverDeterministic(
                receiver,
                opts.vocab_size,
                opts.max_len,
                opts.receiver_emb,
                num_heads=1,
                hidden_size=opts.receiver_hidden,
                num_layers=1,
            )
        else:
            receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
            receiver = core.RnnReceiverDeterministic(
                receiver,
                opts.vocab_size,
                opts.receiver_emb,
                opts.receiver_hidden,
                cell=opts.receiver_cell,
            )

            game = core.SenderReceiverRnnGS(sender, receiver, diff_loss)

        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            diff_loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
        )

    optimizer = torch.optim.Adam(
        [
            dict(params=sender.parameters(), lr=opts.sender_lr),
            dict(params=receiver.parameters(), lr=opts.receiver_lr),
        ]
    )

    loss = game.loss

    intervention = CallbackEvaluator(
        test_loader,
        device=device,
        is_gs=opts.mode == "gs",
        loss=loss,
        var_length=opts.variable_length,
        input_intervention=True,
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True),
            EarlyStopperAccuracy(opts.early_stopping_thr),
            intervention,
        ],
    )

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
