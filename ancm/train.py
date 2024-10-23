# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import argparse
import operator
import pathlib
import json
import uuid
import time
from datetime import timedelta
from collections import defaultdict

import torch.utils.data
from sklearn.metrics import f1_score

import egg.core as core
from egg.core.util import move_to
from egg.zoo.objects_game.features import VectorsLoader

from ancm.trainers import Trainer
from ancm.util import (
    dump_sender_receiver,
    compute_alignment,
    compute_mi_input_msgs,
    compute_top_sim,
    compute_posdis,
    compute_bosdis,
    is_jsonable,
)
from ancm.archs import (
    SenderGS, ReceiverGS,
    loss_gs, SenderReceiverRnnGS,
    SenderReinforce, ReceiverReinforce,
    loss_reinforce, SenderReceiverRnnReinforce
)
from ancm.custom_callbacks import (
    CustomProgressBarLogger,
    LexiconSizeCallback,
    AlignmentCallback,
    TopographicRhoCallback,
    PosDisCallback,
    BosDisCallback
)


def get_params(params):
    parser = argparse.ArgumentParser()

    input_data = parser.add_mutually_exclusive_group()
    input_data.add_argument("--perceptual_dimensions", type=str, default="[4, 4, 4, 4, 4]", help="Number of features for every perceptual dimension")
    input_data.add_argument("--load_data_path", type=str, default=None, help="Path to .npz data file to load")
    parser.add_argument("--n_distractors", type=int, default=3, help="Number of distractor objects for the receiver (default: 3)")
    parser.add_argument("--train_samples", type=float, default=1e5, help="Number of tuples in training data (default: 1e5)")
    parser.add_argument("--validation_samples", type=float, default=1e3, help="Number of tuples in validation data (default: 1e4)")
    parser.add_argument("--test_samples", type=float, default=1e3, help="Number of tuples in test data (default: 1e3)")
    parser.add_argument("--data_seed", type=int, default=42, help="Seed for random creation of train, validation and test tuples (default: 42)")
    parser.add_argument("--erasure_pr", type=float, default=0., help="Probability of erasing a symbol (default: 0.0)")
    parser.add_argument("--no_shuffle", action="store_false", default=True, help="Do not shuffle train data before every epoch (default: False)")
    parser.add_argument("--sender_hidden", type=int, default=50, help="Size of the hidden layer of Sender (default: 50)")
    parser.add_argument("--receiver_hidden", type=int, default=50, help="Size of the hidden layer of Receiver (default: 50)")
    parser.add_argument("--sender_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Sender (default: 10)")
    parser.add_argument("--receiver_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Receiver (default: 10)")
    parser.add_argument("--sender_cell", type=str, default="rnn", help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--receiver_cell", type=str, default="rnn", help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--sender_lr", type=float, default=1e-1, help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument("--receiver_lr", type=float, default=1e-1, help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument("--sender_entropy_coeff", type=float, default=0.01)
    parser.add_argument("--receiver_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=None, help="LR decay (1.0 for no decay)")
    parser.add_argument("--length_cost", type=float, default=1e-2, help="Message length cost")
    parser.add_argument("--temperature", type=float, default=1.0, help="GS temperature for the sender (default: 1.0)")
    parser.add_argument("--mode", type=str, default="gs", help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {gs only at the moment}" "(default: rf)")
    parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate trained model on test data")
    parser.add_argument("--dump_data_folder", type=str, default='data/input_data/', help="Folder where file with dumped data will be created")
    parser.add_argument("--dump_results_folder", type=str, default='runs', help="Folder where file with dumped messages will be created")
    parser.add_argument("--filename", type=str, default=None, help="Filename (no extension)")
    parser.add_argument("--debug", action="store_true", default=False, help="Run egg/objects_game with pdb enabled")
    parser.add_argument("--simple_logging", action="store_true", default=False, help="Use console logger instead of progress bar")
    parser.add_argument("--no_compositionality_metrics", action="store_true", default=False, help="Disable computing topographic rho during training")
    parser.add_argument("--silent", action="store_true", default=False, help="Do not print eval stats during training")

    args = core.init(parser, params)

    check_args(args)
    if not args.silent:
        print(args)

    if args.filename is None:
        args.filename = str(uuid.uuid4())

    return args


def check_args(args):
    args.train_samples, args.validation_samples, args.test_samples = (
        int(args.train_samples), int(args.validation_samples), int(args.test_samples))

    if args.dump_data_folder is not None:
        os.makedirs(os.path.dirname(args.dump_data_folder), exist_ok=True)

    if args.dump_results_folder is not None:
        os.makedirs(os.path.dirname(args.dump_results_folder), exist_ok=True)

    try:
        args.perceptual_dimensions = eval(args.perceptual_dimensions)
    except SyntaxError:
        print("The format of the # of perceptual dimensions param is not correct. "
              "Please change it to string representing a list of int. "
              "Correct format: '[int, ..., int]' ")
        exit(1)

    if args.debug:
        import pdb

        pdb.set_trace()

    args.n_features = len(args.perceptual_dimensions)

    # can't set data loading and data dumping at the same time
    if args.load_data_path:
        args.dump_data_folder = None

    args.dump_results_folder = (
        pathlib.Path(args.dump_results_folder) if args.dump_results_folder is not None else None
    )

    if (not args.evaluate) and args.dump_results_folder:
        print(
            "| WARNING --dump_results_folder was set without --evaluate. Evaluation will not be performed nor any results will be dumped. Please set --evaluate"
        )


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
        shuffle_train_data=opts.no_shuffle,
        dump_data_folder=opts.dump_data_folder,
        load_data_path=opts.load_data_path,
        seed=opts.data_seed)

    train_data, validation_data, test_data = data_loader.get_iterators()

    data_loader.upd_cl_options(opts)

    if opts.mode.lower() == "gs":
        _sender = SenderGS(n_features=data_loader.n_features, n_hidden=opts.sender_hidden)
        _receiver = ReceiverGS(n_features=data_loader.n_features, linear_units=opts.receiver_hidden)
        sender = core.RnnSenderGS(
            _sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature)
        receiver = core.RnnReceiverGS(
            _receiver,
            opts.vocab_size + 1,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell)
        game = SenderReceiverRnnGS(
            sender, receiver, loss_gs,
            opts.max_len, opts.vocab_size, opts.erasure_pr,
            length_cost=opts.length_cost,
            device=device,
            seed=opts.random_seed)
        optimizer = torch.optim.Adam([
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ])

    elif opts.mode.lower() == "rf":
        _sender = SenderReinforce(n_features=data_loader.n_features, n_hidden=opts.sender_hidden)
        _receiver = ReceiverReinforce(n_features=data_loader.n_features, linear_units=opts.receiver_hidden)
        sender = core.RnnSenderReinforce(
            _sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len)
        receiver = core.RnnReceiverReinforce(
            core.ReinforceWrapper(_receiver),
            opts.vocab_size + 1,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell)
        game = SenderReceiverRnnReinforce(
            sender, receiver, loss_reinforce,
            opts.max_len, opts.vocab_size, opts.erasure_pr,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
            length_cost=opts.length_cost,
            device=device,
            seed=opts.random_seed)
        optimizer = torch.optim.RMSprop([
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ])

    else:
        raise NotImplementedError(f"Unknown training mode, {opts.mode}")

    if opts.lr_decay is not None and opts.lr_decay != 1.:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=opts.lr_decay, total_iters=opts.n_epochs)
    else:
        scheduler = None

    if opts.silent:
        callbacks = []
    elif opts.simple_logging:
        callbacks = [core.ConsoleLogger(as_json=True)]
    else:
        callbacks = [
            LexiconSizeCallback(),
            AlignmentCallback(_sender, _receiver, test_data, device, opts.validation_freq, opts.batch_size),
       ]

    if not opts.no_compositionality_metrics:
        callbacks.extend([
            TopographicRhoCallback(opts.perceptual_dimensions),
            PosDisCallback(),
            BosDisCallback(opts.vocab_size),
        ])

    if opts.mode.lower() == "gs":
        callbacks.append(core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1))

    if not opts.silent and not opts.simple_logging:
        callbacks.append(CustomProgressBarLogger(
            n_epochs=opts.n_epochs,
            print_train_metrics=True,
            train_data_len=len(train_data),
            test_data_len=len(validation_data),
            step=opts.validation_freq,
            dump_results_folder=opts.dump_results_folder,
            filename=opts.filename))
 
    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=scheduler,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks)

    t_start = time.monotonic()
    if opts.silent or opts.simple_logging or opts.erasure_pr == 0.:
        trainer.train(n_epochs=opts.n_epochs, second_val=False)
    else:
        trainer.train(n_epochs=opts.n_epochs, second_val=True)
    training_time = timedelta(seconds=time.monotonic()-t_start)
    sec_per_epoch = training_time.seconds / opts.n_epochs
    minutes, seconds = divmod(sec_per_epoch, 60)

    time_total = str(training_time).split('.', maxsplit=1)[0]
    time_per_epoch = f'{int(minutes):02}:{int(seconds):02}'

    if opts.evaluate:
        output_dict = defaultdict(dict)

        # Standard evaluation – same setting as during training
        sender_inputs, messages, receiver_inputs, receiver_outputs, labels = \
            dump_sender_receiver(
                game, test_data, opts.mode == 'gs', apply_noise=opts.erasure_pr != 0,
                variable_length=True, device=device)

        receiver_outputs = move_to(receiver_outputs, device)
        receiver_outputs = torch.stack(receiver_outputs)
        labels = move_to(labels, device)
        labels = torch.stack(labels)

        preds = receiver_outputs.argmax(dim=1) if opts.mode.lower() == 'gs' \
            else receiver_outputs

        accuracy = torch.mean((preds == labels).float()).item() * 100
        f1 =  f1_score(labels, preds, average='micro').item() * 100
        alignment = compute_alignment(
            test_data, _receiver, _sender, device, opts.batch_size)
        top_sim = compute_top_sim(sender_inputs, messages, opts.perceptual_dimensions)
        #pos_dis = 0
        #bos_dis = 0
        #for message in messages:
        pos_dis = compute_posdis(sender_inputs, messages)
        bos_dis = compute_bosdis(sender_inputs, messages, opts.vocab_size)
        #pos_dis = pos_dis/len(messages)
        #bos_dis = bos_dis/len(messages)
    
        output_dict['results']['accuracy'] = accuracy
        output_dict['results']['f1-micro'] = f1
        output_dict['results']['embedding_alignment'] = alignment
        output_dict['results']['topographic_rho'] = top_sim
        output_dict['results']['pos_dis'] = pos_dis
        output_dict['results']['bos_dis'] = bos_dis

        unique_dict = {}
        for elem in sender_inputs:
            target = ""
            for dim in elem:
                target += f"{str(int(dim.item()))}-"
            target = target[:-1]
            if target not in unique_dict:
                unique_dict[target] = True

        mi_result = compute_mi_input_msgs(sender_inputs, messages)
        output_dict['results'].update(mi_result)
        entropy_msg = f"{mi_result['entropy_msg']:.3f}"
        entropy_inp = f"{mi_result['entropy_inp']:.3f}"
        mi = f"{mi_result['mi']:.3f}"
        entropy_inp_dim = f"{[round(x, 3) for x in mi_result['entropy_inp_dim']]}"
        mi_dim = f'{[round(x, 3) for x in mi_result["mi_dim"]]}'
        t_rho = f'{top_sim:.3f}'
        p_dis = f'{pos_dis:.3f}'
        b_dis = f'{bos_dis:.3f}'

        # If we applied noise during training,
        # compute results after disabling noise in the test phase as well
        if opts.erasure_pr != 0:
            sender_inputs2, messages2, receiver_inputs2, \
                receiver_outputs2, labels2 = dump_sender_receiver(
                    game, test_data, opts.mode.lower() == 'gs',
                    apply_noise=opts.erasure_pr == 0,
                    variable_length=True, device=device)

            receiver_outputs2 = move_to(receiver_outputs2, device)
            receiver_outputs2 = torch.stack(receiver_outputs2)
            labels2 = move_to(labels2, device)
            labels2 = torch.stack(labels2)

            preds2 = receiver_outputs2.argmax(dim=1) if opts.mode.lower() == 'gs' \
                else receiver_outputs2
            accuracy2 = torch.mean((preds2 == labels2).float()).item() * 100
            f12 =  f1_score(labels2, preds2, average='micro').item() * 100
            top_sim2 = compute_top_sim(sender_inputs2, messages2, opts.perceptual_dimensions)
            pos_dis2 = compute_posdis(sender_inputs2, messages2)
            bos_dis2 = compute_bosdis(sender_inputs2, messages2, opts.vocab_size)

            output_dict['results-no-noise']['accuracy'] = accuracy2
            output_dict['results-no-noise']['f1-micro'] = f12
            output_dict['results-no-noise']['embedding_alignment'] = alignment
            output_dict['results-no-noise']['topographic_rho'] = top_sim2
            output_dict['results-no-noise']['pos_dis'] = pos_dis2
            output_dict['results-no-noise']['bos_dis'] = bos_dis2

            acc_str = f'{accuracy:.2f} / {accuracy2:.2f}'
            f1_str = f'{f1:.2f} / {f12:.2f}'
            mi_result2 = compute_mi_input_msgs(sender_inputs2, messages2)
            output_dict['results-no-noise'].update(mi_result2)
            entropy_msg += f" / {mi_result2['entropy_msg']:.3f}"
            entropy_inp += f" / {mi_result2['entropy_inp']:.3f}"
            mi += f" / {mi_result2['mi']:.3f}"
            mi_dim2 = f"{[round(x, 3) for x in mi_result2['mi_dim']]}"
            t_rho += f" / {top_sim2:.3f}"
            p_dis += f' / {pos_dis2:.3f}'
            b_dis += f' / {bos_dis2:.3f}'


            if not opts.silent:
                if not opts.simple_logging:
                    print("|")
                print(f"|\033[1m Results (with noise / without noise)\033[0m\n|")
        else:
            acc_str = f'{accuracy:.2f}'
            f1_str = f'{f1:.2f}'
            if not opts.silent:
                print(f"|\n|\033[1m Results\033[0m\n|")

        if not opts.silent:
            align = 23
            print("|" + "H(msg) =".rjust(align), entropy_msg)
            print("|" + "H(target objs) =".rjust(align), entropy_inp)
            print("|" + "I(target objs; msg) =".rjust(align), mi)
            print("|\n| Separately for each object vector dimension")
            if opts.erasure_pr != 0:
                print("|" + "H(target objs) =".rjust(align), entropy_inp_dim)
                print("|" + "I(target objs; msg) =".rjust(align), mi_dim, "(with noise)")
                print("|" + "I(target objs; msg) =".rjust(align), mi_dim2, "(no noise)")
            else:
                print("|" + "H(target objs) =".rjust(align) + entropy_inp_dim, "(for each dimension)")
                print("|" + "I(target objs; msg) =".rjust(align), mi_dim, "(for each dimension)")
            print('|')
            print("|" + "Accuracy:".rjust(align), acc_str)
            print("|" + "F1 (micro):".rjust(align), f1_str)
            print("|")
            print("|" + "Embedding alignment:".rjust(align) + f" {alignment:.2f}")
            print("|" + "Topographic rho:".rjust(align) + f" {t_rho}")
            print("|" + "PosDis:".rjust(align) + f" {p_dis}")
            print("|" + "BosDis:".rjust(align) + f" {b_dis}")

        if opts.dump_results_folder:
            opts.dump_results_folder.mkdir(exist_ok=True)

            messages_dict = {}

            msg_dict = defaultdict(int)
            for sender_input, message, receiver_input, receiver_output, label \
                    in zip(sender_inputs, messages, receiver_inputs, receiver_outputs, labels):
                target_vec = ','.join([str(int(x)) for x in sender_input.tolist()])
                message = ','.join([str(int(x)) for x in message.tolist()])
                candidate_vex = [','.join([str(int(x)) for x in candidate])
                                 for candidate in receiver_input.tolist()]
                message_log = {
                    'target_vec': target_vec,
                    'candidate_vex': candidate_vex,
                    'message': message}
                if opts.erasure_pr != 0:
                    message_log['message_no_noise'] = None
                message_log['label'] = label.item()

                m_key = f'{target_vec}#' + ';'.join(candidate_vex)
                messages_dict[m_key] = message_log
                msg_dict[message] += 1

            sorted_msgs = sorted(msg_dict.items(), key=operator.itemgetter(1), reverse=True)

            if opts.erasure_pr != 0.:
                msg_dict2 = defaultdict(int)
                for sender_input, message, receiver_input, receiver_output, label \
                        in zip(sender_inputs2, messages2, receiver_inputs2, receiver_outputs2, labels2):
                    target_vec = ','.join([str(int(x)) for x in sender_input.tolist()])
                    candidate_vex = [','.join([str(int(c)) for c in candidate])
                                     for candidate in receiver_input.tolist()]
                    message = ','.join([str(int(x)) for x in message.tolist()])

                    m_key = f'{target_vec}#' + ';'.join(candidate_vex)
                    messages_dict[m_key]['message_no_noise'] = message
                    msg_dict2[message] += 1

                sorted_msgs2 = sorted(msg_dict2.items(), key=operator.itemgetter(1), reverse=True)

            lexicon_size = str(len(msg_dict.keys())) if opts.erasure_pr == 0 \
                else f'{len(msg_dict.keys())} / {len(msg_dict2.keys())}'
            if not opts.silent:
                print("|")
                print("|" + "Unique target objects:".rjust(align), len(unique_dict.keys()))
                print("|" + "Lexicon size:".rjust(align), lexicon_size)

            output_dict['results']['unique_targets'] = len(unique_dict.keys())
            output_dict['results']['unique_msg'] = len(msg_dict.keys())
            if opts.erasure_pr != 0:
                output_dict['results']['unique_msg_no_noise'] = len(msg_dict2.keys())
            output_dict['results']['embedding_alignment'] = alignment
            output_dict['messages'] = [v for v in messages_dict.values()]
            output_dict['message_counts'] = sorted_msgs
            if opts.erasure_pr != 0:
                output_dict['message_counts_no_noise'] = sorted_msgs2
                output_dict['erased_symbol'] = opts.vocab_size
            opts_dict = {k: v for k, v in vars(opts).items() if is_jsonable(v)}
            output_dict['opts'] = opts_dict
            output_dict['training_time'] = {
                'total': time_total,
                'per_epoch': time_per_epoch}

            with open(opts.dump_results_folder / f'{opts.filename}-results.json', 'w') as f:
                json.dump(output_dict, f, indent=4)

            if not opts.silent:
                print(f"| Results saved to {opts.dump_results_folder/opts.filename}-results.json")

    if not opts.silent:
        print('| Total training time:', time_total)
        print('| Training time per epoch:', time_per_epoch)

if __name__ == "__main__":

    import sys

    main(sys.argv[1:])
