# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compo_vs_generalization.archs import (
    Freezer,
    NonLinearReceiver,
    PlusOneWrapper,
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=4, help="")
    parser.add_argument("--n_values", type=int, default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=100)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument(
        "--baseline", type=str, choices=["no", "mean", "builtin"], default="mean"
    )
    parser.add_argument(
        "--density_data", type=int, default=0, help="no sampling if equal 0"
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )

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
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, generalization=False):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.test_generalization:
            acc, acc_or, loss = 0.0, 0.0, 0.0

            for attr in range(self.n_attributes):
                zero_index = torch.nonzero(sender_input[:, attr, 0]).squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat(
                    [masked_input[:, :attr, :], masked_input[:, attr + 1 :, :]], dim=1
                )
                no_attribute_output = torch.cat(
                    [masked_output[:, :attr, :], masked_output[:, attr + 1 :, :]], dim=1
                )

                n_attributes = self.n_attributes - 1
                attr_acc = (
                    (
                        (
                            no_attribute_output.argmax(dim=-1)
                            == no_attribute_input.argmax(dim=-1)
                        ).sum(dim=1)
                        == n_attributes
                    )
                    .float()
                    .mean()
                )
                acc += attr_acc

                attr_acc_or = (
                    (
                        no_attribute_output.argmax(dim=-1)
                        == no_attribute_input.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                )
                acc_or += attr_acc_or
                labels = no_attribute_input.argmax(dim=-1).view(
                    masked_size * n_attributes
                )
                predictions = no_attribute_output.view(
                    masked_size * n_attributes, self.n_values
                )
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc /= self.n_attributes
            acc_or /= self.n_attributes
        else:
            acc = (
                torch.sum(
                    (
                        receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
                    ).detach(),
                    dim=1,
                )
                == self.n_attributes
            ).float()
            acc_or = (
                receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
            ).float()

            receiver_output = receiver_output.view(
                batch_size * self.n_attributes, self.n_values
            )
            labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
            loss = (
                F.cross_entropy(receiver_output, labels, reduction="none")
                .view(batch_size, self.n_attributes)
                .mean(dim=-1)
            )

        return loss, {"acc": acc, "acc_or": acc_or}


def main(params):
    import copy

    opts = get_params(params)
    device = opts.device

    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    if opts.density_data > 0:
        sampled_data = select_subset_V2(
            full_data, opts.density_data, opts.n_attributes, opts.n_values
        )
        full_data = copy.deepcopy(sampled_data)

    train, generalization_holdout = split_holdout(full_data)
    train, uniform_holdout = split_train_test(train, 0.1)

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

    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=len(validation))

    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell in ["lstm", "rnn", "gru"]:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
    else:
        raise ValueError(f"Unknown receiver cell, {opts.receiver_cell}")

    if opts.sender_cell in ["lstm", "rnn", "gru"]:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            cell=opts.sender_cell,
        )
    else:
        raise ValueError(f"Unknown sender cell, {opts.sender_cell}")

    sender = PlusOneWrapper(sender)
    loss = DiffLoss(opts.n_attributes, opts.n_values)

    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
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

    loaders = []
    loaders.append(
        (
            "generalization hold out",
            generalization_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values, generalization=True),
        )
    )
    loaders.append(
        (
            "uniform holdout",
            uniform_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values),
        )
    )

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
            metrics_evaluator,
            holdout_evaluator,
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)

    last_epoch_interaction = early_stopper.validation_stats[-1][1]
    validation_acc = last_epoch_interaction.aux["acc"].mean()

    uniformtest_acc = holdout_evaluator.results["uniform holdout"]["acc"]

    # Train new agents
    if validation_acc > 0.99:

        def _set_seed(seed):
            import random

            import numpy as np

            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        core.get_opts().preemptable = False
        core.get_opts().checkpoint_path = None

        # freeze Sender and probe how fast a simple Receiver will learn the thing
        def retrain_receiver(receiver_generator, sender):
            receiver = receiver_generator()
            game = core.SenderReceiverRnnReinforce(
                sender,
                receiver,
                loss,
                sender_entropy_coeff=0.0,
                receiver_entropy_coeff=0.0,
            )
            optimizer = torch.optim.Adam(receiver.parameters(), lr=opts.lr)
            early_stopper = EarlyStopperAccuracy(
                opts.early_stopping_thr, validation=True
            )

            trainer = core.Trainer(
                game=game,
                optimizer=optimizer,
                train_data=train_loader,
                validation_data=validation_loader,
                callbacks=[early_stopper, Evaluator(loaders, opts.device, freq=0)],
            )
            trainer.train(n_epochs=opts.n_epochs // 2)

            accs = [x[1]["acc"] for x in early_stopper.validation_stats]
            return accs

        frozen_sender = Freezer(copy.deepcopy(sender))

        def gru_receiver_generator():
            return core.RnnReceiverDeterministic(
                Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim),
                opts.vocab_size + 1,
                opts.receiver_emb,
                hidden_size=opts.receiver_hidden,
                cell="gru",
            )

        def small_gru_receiver_generator():
            return core.RnnReceiverDeterministic(
                Receiver(n_hidden=100, n_outputs=n_dim),
                opts.vocab_size + 1,
                opts.receiver_emb,
                hidden_size=100,
                cell="gru",
            )

        def tiny_gru_receiver_generator():
            return core.RnnReceiverDeterministic(
                Receiver(n_hidden=50, n_outputs=n_dim),
                opts.vocab_size + 1,
                opts.receiver_emb,
                hidden_size=50,
                cell="gru",
            )

        def nonlinear_receiver_generator():
            return NonLinearReceiver(
                n_outputs=n_dim,
                vocab_size=opts.vocab_size + 1,
                max_length=opts.max_len,
                n_hidden=opts.receiver_hidden,
            )

        for name, receiver_generator in [
            ("gru", gru_receiver_generator),
            ("nonlinear", nonlinear_receiver_generator),
            ("tiny_gru", tiny_gru_receiver_generator),
            ("small_gru", small_gru_receiver_generator),
        ]:

            for seed in range(17, 17 + 3):
                _set_seed(seed)
                accs = retrain_receiver(receiver_generator, frozen_sender)
                accs += [1.0] * (opts.n_epochs // 2 - len(accs))
                auc = sum(accs)
                print(
                    json.dumps(
                        {
                            "mode": "reset",
                            "seed": seed,
                            "receiver_name": name,
                            "auc": auc,
                        }
                    )
                )

    print("---End--")

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
