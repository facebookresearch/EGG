import os
import argparse
import torch
import random
import numpy as np

from reformat_visa import reformat, get_filename
from features import VectorsLoader
from archs import (
    SenderGS, ReceiverGS, loss_gs,
    SenderReinforce, ReceiverReinforce, loss_reinforce
)

from custom_callbacks import CustomProgressBarLogger

import egg.core as core
from egg.core.util import move_to
from egg.zoo.objects_game.util import (
    compute_baseline_accuracy,
    compute_mi_input_msgs,
    dump_sender_receiver,
    entropy,
    mutual_info,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_distractors", type=int, default=4, help="Number of distractor objects for the receiver (default: 4)")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples (sets of target+distractor concepts) per target concept")
    parser.add_argument("--category", type=str, default=None, help="Category of concepts to be included in the dataset. If category isn't specified, all concepts are included")
    parser.add_argument("--load_data_path", type=str, default=None, help="Path to .npz data file to load")
    parser.add_argument("--sender_hidden", type=int, default=50, help="Size of the hidden layer of Sender (default: 50)")
    parser.add_argument("--receiver_hidden", type=int, default=50, help="Size of the hidden layer of Receiver (default: 50)")
    parser.add_argument("--sender_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Sender (default: 10)")
    parser.add_argument("--receiver_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Receiver (default: 10)")
    parser.add_argument("--sender_cell", type=str, default="lstm", help="Type of the cell used for Sender {rnn, gru, lstm} (default: lstm)")
    parser.add_argument("--receiver_cell", type=str, default="lstm", help="Type of the cell used for Receiver {rnn, gru, lstm} (default: lstm)")
    parser.add_argument("--sender_lr", type=float, default=1e-1, help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument("--receiver_lr", type=float, default=1e-1, help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument("--lr_decay", type=float, default=1., help="LR decay (1.0 for no decay)")
    parser.add_argument("--temperature", type=float, default=1.0, help="GS temperature for the sender (default: 1.0)")
    parser.add_argument("--length_cost", type=float, default=0., help="Message length cost (e.g. 1e-2)")
    parser.add_argument("--mode", type=str, default="rf", help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training (default: rf)")
    parser.add_argument("--output_json", action="store_true", default=False, help="If set, egg will output validation stats in json format (default: False)")
    parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate trained model on test data")
    parser.add_argument("--metrics_step", type=int, default=1, help="Specify the numper of epochs between which metrics are printed")
    parser.add_argument("--dump_msg_folder", type=str, default=None, help="Folder where file with dumped messages will be created")
    parser.add_argument("--debug", action="store_true", default=False, help="Run egg/objects_game with pdb enabled")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = core.init(parser)

    # if load_data_path is not provided, generate the data and load it
    if args.load_data_path is None:
        args.load_data_path = get_filename(args.n_distractors, args.n_samples, args.category, True)
        if not os.path.isfile(args.load_data_path):
            reformat(args.n_distractors, args.n_samples, args.category)

    # perceptual_dimensions default val
    dataset = np.load(args.load_data_path)
    n_attributes = dataset[dataset.files[0]].shape[2]
    perceptual_dimensions = [2 for _ in range(n_attributes)]
    args.perceptual_dimensions = perceptual_dimensions

    # if dump_msg_folder is not provided, fix a folder
    if args.dump_msg_folder is None:
        parent_folder = f'runs/run-{args.category}-{args.n_distractors+1}d-{args.n_samples}s-{args.n_epochs}ep-{args.vocab_size}words'
        msg_dir = os.makedirs(os.path.join(parent_folder, 'msg'), exist_ok=True)
        args.dump_msg_folder = msg_dir

    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = VectorsLoader(
        batch_size=args.batch_size,
        shuffle_train_data=True,
        load_data_path=args.load_data_path,
        seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data, validation_data, test_data = data_loader.get_iterators()

    if args.mode.lower() == "gs":
        sender = SenderGS(n_features=data_loader.n_features, n_hidden=args.sender_hidden)
        receiver = ReceiverGS(n_features=data_loader.n_features, linear_units=args.receiver_hidden)
        sender = core.RnnSenderGS(
            sender,
            args.vocab_size,
            args.sender_embedding,
            args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len,
            temperature=args.temperature)
        receiver = core.RnnReceiverGS(
            receiver,
            args.vocab_size,
            args.receiver_embedding,
            args.receiver_hidden,
            cell=args.receiver_cell)
        game = core.SenderReceiverRnnGS(sender, receiver, loss_gs)
        optimizer = torch.optim.Adam([
            {"params": game.sender.parameters(), "lr": args.sender_lr},
            {"params": game.receiver.parameters(), "lr": args.receiver_lr},
        ])

    elif args.mode.lower() == 'rf':
        sender = SenderReinforce(n_features=data_loader.n_features, n_hidden=args.sender_hidden)
        receiver = ReceiverReinforce(n_features=data_loader.n_features, linear_units=args.receiver_hidden)
        sender = core.RnnSenderReinforce(
            sender,
            args.vocab_size,
            args.sender_embedding,
            args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len)
        receiver = core.RnnReceiverReinforce(
            core.ReinforceWrapper(receiver),
            args.vocab_size,
            args.receiver_embedding,
            args.receiver_hidden,
            cell=args.receiver_cell)
        game = core.SenderReceiverRnnReinforce(
            sender, receiver, loss_reinforce,
            sender_entropy_coeff=0.01,
            receiver_entropy_coeff=0.001,
            length_cost=args.length_cost)
        optimizer = torch.optim.RMSprop([
            {"params": game.sender.parameters(), "lr": args.sender_lr},
            {"params": game.receiver.parameters(), "lr": args.receiver_lr},
        ])

    else:
        raise NotImplementedError(f"Unknown training mode, {args.mode}")

    if args.lr_decay != 1.:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.lr_decay, total_iters=args.n_epochs)
    else:
        scheduler = None

    callbacks = [
        CustomProgressBarLogger(
            n_epochs=args.n_epochs,
            print_train_metrics=True,
            train_data_len=len(train_data),
            test_data_len=len(validation_data),
            step=args.metrics_step)
    ]
    if args.mode.lower() == "gs":
        callbacks.append(core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1))
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=scheduler,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks)

    trainer.train(n_epochs=args.n_epochs)

    if args.evaluate:
        is_gs = args.mode == "gs"
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

        if args.mode.lower() == 'gs':
            tensor_accuracy = receiver_outputs.argmax(dim=1) == labels
        else:
            tensor_accuracy = receiver_outputs == labels
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

        if args.dump_msg_folder:
            args.dump_msg_folder.mkdir(exist_ok=True)
            msg_dict = {}

            output_msg = (
                f"messages_{args.perceptual_dimensions}_vocab_{args.vocab_size}"
                f"_maxlen_{args.max_len}_bsize_{args.batch_size}"
                f"_n_distractors_{args.n_distractors}_n_samples_{args.n_samples}"
                f"_train_size_{args.train_samples}length_cost_{args.length_cost}"
                f"_valid_size_{args.validation_samples}_test_size_{args.test_samples}"
                f"_slr_{args.sender_lr}_rlr_{args.receiver_lr}_shidden_{args.sender_hidden}"
                f"_rhidden_{args.receiver_hidden}_semb_{args.sender_embedding}"
                f"_remb_{args.receiver_embedding}_mode_{args.mode}"
                f"_lr_decay_{args.lr_decay}_seed_{args.seed}"
                f"_scell_{args.sender_cell}_rcell_{args.receiver_cell}.msg"
            )

            output_file = args.dump_msg_folder / output_msg
            with open(output_file, "w") as f:
                f.write(f"{args}\n")
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


if __name__ == '__main__':
    main()
