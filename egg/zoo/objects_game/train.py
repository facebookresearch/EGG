# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse
import operator
import pathlib

import numpy as np
import torch.nn.functional as F
import torch.utils.data

import egg.core as core
from egg.core.util import move_to
from egg.zoo.objects_game.archs import Receiver, Sender
from egg.zoo.objects_game.features import VectorsLoader
from egg.zoo.objects_game.util import (
    compute_baseline_accuracy,
    compute_mi_input_msgs,
    dump_sender_receiver,
    entropy,
    mutual_info,
)


def get_params(params):
    parser = argparse.ArgumentParser()

    input_data = parser.add_mutually_exclusive_group()
    input_data.add_argument(
        "--perceptual_dimensions",
        type=str,
        default="[4, 4, 4, 4, 4]",
        help="Number of features for every perceptual dimension",
    )
    input_data.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path to .npz data file to load",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=3,
        help="Number of distractor objects for the receiver (default: 3)",
    )
    parser.add_argument(
        "--train_samples",
        type=float,
        default=1e5,
        help="Number of tuples in training data (default: 1e6)",
    )
    parser.add_argument(
        "--validation_samples",
        type=float,
        default=1e3,
        help="Number of tuples in validation data (default: 1e4)",
    )
    parser.add_argument(
        "--test_samples",
        type=float,
        default=1e3,
        help="Number of tuples in test data (default: 1e3)",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=111,
        help="Seed for random creation of train, validation and test tuples (default: 111)",
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
        default=50,
        help="Size of the hidden layer of Sender (default: 50)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 50)",
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
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Sender's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Receiver's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {gs only at the moment}"
        "(default: rf)",
    )

    parser.add_argument(
        "--output_json",
        action="store_true",
        default=False,
        help="If set, egg will output validation stats in json format (default: False)",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate trained model on test data",
    )

    parser.add_argument(
        "--dump_data_folder",
        type=str,
        default=None,
        help="Folder where file with dumped data will be created",
    )
    parser.add_argument(
        "--dump_msg_folder",
        type=str,
        default=None,
        help="Folder where file with dumped messages will be created",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run egg/objects_game with pdb enabled",
    )

    args = core.init(parser, params)

    check_args(args)
    print(args)

    return args


def check_args(args):
    args.train_samples, args.validation_samples, args.test_samples = (
        int(args.train_samples),
        int(args.validation_samples),
        int(args.test_samples),
    )

    try:
        args.perceptual_dimensions = eval(args.perceptual_dimensions)
    except SyntaxError:
        print(
            "The format of the # of perceptual dimensions param is not correct. Please change it to string representing a list of int. Correct format: '[int, ..., int]' "
        )
        exit(1)

    if args.debug:
        import pdb

        pdb.set_trace()

    args.n_features = len(args.perceptual_dimensions)

    # can't set data loading and data dumping at the same time
    assert not (
        args.load_data_path and args.dump_data_folder
    ), "Cannot set folder to dump data while setting path to vectors to be loaded. Are you trying to dump the same vectors that you are loading?"

    args.dump_msg_folder = (
        pathlib.Path(args.dump_msg_folder) if args.dump_msg_folder is not None else None
    )

    if (not args.evaluate) and args.dump_msg_folder:
        print(
            "| WARNING --dump_msg_folder was set without --evaluate. Evaluation will not be performed nor any results will be dumped. Please set --evaluate"
        )


def loss(
    _sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input
):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {"acc": acc}


def main(params):
    opts = get_params(params)

    device = torch.device("cuda" if opts.cuda else "cpu")

    data_loader = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=opts.shuffle_train_data,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=opts.load_data_path,
        seed=opts.data_seed,
    )

    train_data, validation_data, test_data = data_loader.get_iterators()

    data_loader.upd_cl_options(opts)

    if opts.max_len > 1:
        baseline_msg = 'Cannot yet compute "smart" baseline value for messages of length greater than 1'
    else:
        baseline_msg = (
            f"\n| Baselines measures with {opts.n_distractors} distractors and messages of max_len = {opts.max_len}:\n"
            f"| Dummy random baseline: accuracy = {1 / (opts.n_distractors + 1)}\n"
        )
        if -1 not in opts.perceptual_dimensions:
            baseline_msg += f'| "Smart" baseline with perceptual_dimensions {opts.perceptual_dimensions} = {compute_baseline_accuracy(opts.n_distractors, opts.max_len, *opts.perceptual_dimensions)}\n'
        else:
            baseline_msg += f'| Data was loaded froman external file, thus no perceptual_dimension vector was provided, "smart baseline" cannot be computed\n'

    print(baseline_msg)

    sender = Sender(n_features=data_loader.n_features, n_hidden=opts.sender_hidden)

    receiver = Receiver(
        n_features=data_loader.n_features, linear_units=opts.receiver_hidden
    )

    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )

        receiver = core.RnnReceiverGS(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

        game = core.SenderReceiverRnnGS(sender, receiver, loss)
    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    optimizer = torch.optim.Adam(
        [
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ]
    )
    callbacks = [core.ConsoleLogger(as_json=True)]
    if opts.mode.lower() == "gs":
        callbacks.append(core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1))
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)

    if opts.evaluate:
        is_gs = opts.mode == "gs"
        (
            sender_inputs,
            messages,
            receiver_inputs,
            receiver_outputs,
            labels,
        ) = dump_sender_receiver(
            game, test_data, is_gs, variable_length=True, device=device
        )

        receiver_outputs = move_to(receiver_outputs, device)
        labels = move_to(labels, device)

        receiver_outputs = torch.stack(receiver_outputs)
        labels = torch.stack(labels)

        tensor_accuracy = receiver_outputs.argmax(dim=1) == labels
        accuracy = torch.mean(tensor_accuracy.float()).item()

        unique_dict = {}

        for elem in sender_inputs:
            target = ""
            for dim in elem:
                target += f"{str(int(dim.item()))}-"
            target = target[:-1]
            if target not in unique_dict:
                unique_dict[target] = True

        print(f"| Accuracy on test set: {accuracy}")

        compute_mi_input_msgs(sender_inputs, messages)

        print(f"entropy sender inputs {entropy(sender_inputs)}")
        print(f"mi sender inputs msgs {mutual_info(sender_inputs, messages)}")

        if opts.dump_msg_folder:
            opts.dump_msg_folder.mkdir(exist_ok=True)
            msg_dict = {}

            output_msg = (
                f"messages_{opts.perceptual_dimensions}_vocab_{opts.vocab_size}"
                f"_maxlen_{opts.max_len}_bsize_{opts.batch_size}"
                f"_n_distractors_{opts.n_distractors}_train_size_{opts.train_samples}"
                f"_valid_size_{opts.validation_samples}_test_size_{opts.test_samples}"
                f"_slr_{opts.sender_lr}_rlr_{opts.receiver_lr}_shidden_{opts.sender_hidden}"
                f"_rhidden_{opts.receiver_hidden}_semb_{opts.sender_embedding}"
                f"_remb_{opts.receiver_embedding}_mode_{opts.mode}"
                f"_scell_{opts.sender_cell}_rcell_{opts.receiver_cell}.msg"
            )

            output_file = opts.dump_msg_folder / output_msg
            with open(output_file, "w") as f:
                f.write(f"{opts}\n")
                for (
                    sender_input,
                    message,
                    receiver_input,
                    receiver_output,
                    label,
                ) in zip(
                    sender_inputs, messages, receiver_inputs, receiver_outputs, labels
                ):
                    sender_input = ",".join(map(str, sender_input.tolist()))
                    message = ",".join(map(str, message.tolist()))
                    distractors_list = receiver_input.tolist()
                    receiver_input = "; ".join(
                        [",".join(map(str, elem)) for elem in distractors_list]
                    )
                    if is_gs:
                        receiver_output = receiver_output.argmax()
                    f.write(
                        f"{sender_input} -> {receiver_input} -> {message} -> {receiver_output} (label={label.item()})\n"
                    )

                    if message in msg_dict:
                        msg_dict[message] += 1
                    else:
                        msg_dict[message] = 1

                sorted_msgs = sorted(
                    msg_dict.items(), key=operator.itemgetter(1), reverse=True
                )
                f.write(
                    f"\nUnique target vectors seen by sender: {len(unique_dict.keys())}\n"
                )
                f.write(f"Unique messages produced by sender: {len(msg_dict.keys())}\n")
                f.write(f"Messagses: 'msg' : msg_count: {str(sorted_msgs)}\n")
                f.write(f"\nAccuracy: {accuracy}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
