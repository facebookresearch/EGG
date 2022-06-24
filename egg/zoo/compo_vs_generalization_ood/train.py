# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
this script is a modification of `compo_vs_generalization/train.py`
"""

import argparse
import torch
from torch.utils.data import DataLoader

from egg import core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compo_vs_generalization.train import DiffLoss
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics
import egg.zoo.compo_vs_generalization_ood.archs


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=4, help="")
    parser.add_argument("--n_values", type=int, default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=100)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument(
        "--hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )
    parser.add_argument("--sender", type=str)
    parser.add_argument("--receiver", type=str)
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
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )

    args = core.init(arg_parser=parser, params=params)
    return args


def get_data(opts):
    """
    creating all possible ordered pairs for given n_values.

    Splitting the pairs into:
    generalization_holdout ... all pairs with a zero, not including three pairs:
                               [(0,0), (0,1), (1,0)]
    uniform_holdout ... 10% of pairs without a zero (e.g. (42,13), (13,1), ...)
    train ... 90% of pairs without a zero plus three pairs with a zero
              (e.g. (0,0), (0,1), (1,0), (23,1), (2,43), ...)
    """
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    train, generalization_holdout = split_holdout(full_data)
    train, uniform_holdout = split_train_test(train, 0.1)
    assert opts.n_attributes == 2
    additional_training_pairs = [(0, 0), (0, 1), (1, 0)]
    train = additional_training_pairs + train
    for pair in additional_training_pairs[1:]:
        # (0 , 0) is not in generalization_holdout
        generalization_holdout.remove(pair)
    return full_data, train, uniform_holdout, generalization_holdout


def main(params):
    opts = get_params(params)
    print(opts)

    full_data, train, uniform_holdout, generalization_holdout = get_data(opts)

    generalization_holdout, train, uniform_holdout, full_data = [
        one_hotify(x, opts.n_attributes, opts.n_values)
        for x in [generalization_holdout, train, uniform_holdout, full_data]
    ]

    train, validation = ScaledDataset(train, opts.data_scaler), ScaledDataset(train, 1)

    generalization_holdout, uniform_holdout, full_data = (
        ScaledDataset(generalization_holdout),
        ScaledDataset(uniform_holdout),
        ScaledDataset(full_data),
    )
    generalization_holdout_loader, uniform_holdout_loader, full_data_loader = [
        DataLoader(x, batch_size=opts.batch_size)
        for x in [generalization_holdout, uniform_holdout, full_data]
    ]

    train_loader = DataLoader(train, batch_size=opts.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=len(validation))

    loss = DiffLoss(opts.n_attributes, opts.n_values)

    sender = getattr(egg.zoo.compo_vs_generalization_ood.archs, opts.sender)(opts)
    receiver = getattr(egg.zoo.compo_vs_generalization_ood.archs, opts.receiver)(opts)

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=core.baselines.MeanBaseline,
    )
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    metrics_evaluator = Metrics(
        validation.examples,
        opts.device,
        opts.n_attributes,
        opts.n_values,
        opts.vocab_size + 1,
        freq=opts.stats_freq,
    )

    metrics_evaluator_generalization_holdout = Metrics(
        generalization_holdout.examples,
        opts.device,
        opts.n_attributes,
        opts.n_values,
        opts.vocab_size + 1,
        freq=opts.stats_freq,
    )

    loaders = []
    loaders.append(
        (
            "generalization hold out",
            generalization_holdout_loader,
            # DiffLoss(opts.n_attributes, opts.n_values, generalization=True),
            # we don't want to ignore zeros:
            DiffLoss(opts.n_attributes, opts.n_values, generalization=False),
        )
    )
    loaders.append(
        (
            "uniform holdout",
            uniform_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values),
        )
    )

    holdout_evaluator = Evaluator(loaders, opts.device, freq=1)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            # print validation (i.e. unscaled training data) loss:
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
            # print compositionality metrics at the end of training
            #   (validation, i.e, unscaled training data):
            metrics_evaluator,
            # print compositionality metrics at the end of training (holdout data):
            metrics_evaluator_generalization_holdout,
            # print generalization and uniform holdout accuracies at each epoch:
            holdout_evaluator,
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
