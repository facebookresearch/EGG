# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import numpy as np
import torch.nn.functional as F
import torch.utils.data
import pandas as pd

import sys
sys.path.append("C:\\Users\\franc\\Documents\\GitHub\\ANCM_EGG\\egg")

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.core import LoggingStrategy
from egg.zoo.adjusted_channel_game.archs import Receiver, Sender
from egg.zoo.adjusted_channel_game.features import OneHotLoader
from egg.zoo.adjusted_channel_game.reformat_visa import reformat


def get_params(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path to .npz data file to load",
    )

    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=1000,
        help="Number of batches per epoch (default: 1000)",
    )
    parser.add_argument(
        "--shuffle_train_data",
        action="store_true",
        default=False,
        help="Shuffle train data before every epoch (default: False)",
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
        "--receiver_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of receiver. Only in reinforce (default: 1)",
    )
    parser.add_argument(
        "--sender_num_layers",
        type=int,
        default=1,
        help="Number hidden layers of receiver. Only in reinforce (default: 1)",
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
        default="lstm",
        help="Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: lstm)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="lstm",
        help="Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: lstm)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="The entropy regularisation coefficient for Sender (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_entropy_coeff",
        type=float,
        default=1e-1,
        help="The entropy regularisation coefficient for Receiver (default: 1e-1)",
    )

    parser.add_argument(
        "--probs",
        type=str,
        default="uniform",
        help="Prior distribution over the concepts (default: uniform)",
    )
    parser.add_argument(
        "--length_cost",
        type=float,
        default=0.0,
        help="Penalty for the message length, each symbol would before <EOS> would be "
        "penalized by this cost (default: 0.0)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="model",
        help="Name for your checkpoint (default: model)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.9999,
        help="Early stopping threshold on accuracy (default: 0.9999)",
    )

    parser.add_argument(
        "--dump_data_folder",
        type=str,
        default=None,
        help="Folder where file with dumped data will be created",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=111,
        help="Seed for random creation of train, validation and test tuples (default: 111)",
    )

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(
        receiver_output, sender_input.argmax(dim=1), reduction="none"
    )
    return loss, {"acc": acc}


def dump(game, n_features, device, gs_mode):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    interaction = core.dump_interactions(
        game, dataset, gs=gs_mode, device=device, variable_length=True
    )

    unif_acc = 0.0

    for i in range(interaction.size):
        sender_input = interaction.sender_input[i]
        message = interaction.message[i]
        receiver_output = interaction.receiver_output[i]

        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        print(
            f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}',
            flush=True,
        )

    unif_acc /= n_features


def main(params):
    opts = get_params(params)
    print(opts, flush=True)

    device = torch.device("cuda" if opts.cuda else "cpu")
    opts.max_len -= 1

    train_list, valid_list, test_list = reformat()
    n_features = len(train_list[0]) # check how many features there are

    train_data = OneHotLoader(
        seed= opts.random_seed,
        one_hot_list=train_list,
        batches_per_epoch=opts.batches_per_epoch,
        batch_size=opts.batch_size
    )

    validation_data = OneHotLoader(
        seed= opts.random_seed,
        one_hot_list=valid_list,
        batches_per_epoch=opts.batches_per_epoch,
        batch_size=opts.batch_size
    )
    
    sender = Sender(n_features=n_features, n_hidden=opts.sender_hidden)

    sender = core.RnnSenderReinforce(
        sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len,
        num_layers=opts.sender_num_layers,
    )
    
    receiver = Receiver(n_features=n_features, n_hidden=opts.receiver_hidden)
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell,
        num_layers=opts.receiver_num_layers,
    )

    empty_logger = LoggingStrategy.minimal()
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        train_logging_strategy=empty_logger,
        length_cost=opts.length_cost,
    )

    optimizer = core.build_optimizer(game.parameters())

    callbacks = [
        EarlyStopperAccuracy(opts.early_stopping_thr),
        core.ConsoleLogger(as_json=True, print_train_loss=True),
    ]

    if opts.checkpoint_dir:
        checkpoint_name = f"{opts.name}_vocab{opts.vocab_size}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}"
        callbacks.append(
            core.CheckpointSaver(
                checkpoint_path=opts.checkpoint_dir, prefix=checkpoint_name
            )
        )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=opts.n_epochs)

    game.logging_strategy = LoggingStrategy.maximal()  # now log everything
    dump(trainer.game, n_features, device, False)
    core.close()



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
