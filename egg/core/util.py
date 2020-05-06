# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Iterable, List, Optional, Any

import sys
import random
import argparse
import torch
import numpy as np

from collections import defaultdict

common_opts = None
optimizer = None
summary_writer = None


def _populate_cl_params(arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    arg_parser.add_argument('--random_seed', type=int, default=None,
                        help='Set random seed')
    # trainer params
    arg_parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Where the checkpoints are stored')
    arg_parser.add_argument('--preemptable', default=False,
                            action='store_true',
                            help='If the flag is set, Trainer would always try to initialise itself from a checkpoint')

    arg_parser.add_argument('--checkpoint_freq', type=int, default=0,
                        help='How often the checkpoints are saved')
    arg_parser.add_argument('--validation_freq', type=int, default=1,
                        help='The validation would be run every `validation_freq` epochs')
    arg_parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    arg_parser.add_argument('--load_from_checkpoint', type=str, default=None,
                        help='If the parameter is set, model, trainer, and optimizer states are loaded from the '
                             'checkpoint (default: None)')
    # cuda setup
    arg_parser.add_argument('--no_cuda', default=False, help='disable cuda',
                        action='store_true')
    # dataset
    arg_parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')

    # optimizer
    arg_parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use [adam, sgd, adagrad] (default: adam)')
    arg_parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')

    # Channel parameters
    arg_parser.add_argument('--vocab_size', type=int, default=10,
                        help='Number of symbols (terms) in the vocabulary (default: 10)')
    arg_parser.add_argument('--max_len', type=int, default=1,
                        help='Max length of the sequence (default: 1)')

    # Setting up tensorboard
    arg_parser.add_argument('--tensorboard', default=False, help='enable tensorboard',
                            action='store_true')
    arg_parser.add_argument('--tensorboard_dir', type=str, default='runs/',
                            help='Path for tensorboard log')

    return arg_parser


def _get_params(arg_parser: argparse.ArgumentParser, params: List[str]) -> argparse.Namespace:
    args = arg_parser.parse_args(params)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # just to avoid confusion and be consistent
    args.no_cuda = not args.cuda
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def init(arg_parser:Optional[argparse.ArgumentParser] = None, params:Optional[List[str]] = None) -> argparse.Namespace:
    """
    Should be called before any code using egg; initializes the common components, such as
    seeding logic etc.

    :param arg_parser: An instance of argparse.ArgumentParser that is pre-populated if game-specific arguments.
        `init` would add the commonly used arguments and parse the CL parameters. This allows us to easily obtain
        commonly used parameters and have a full list of parameters obtained by a `--help` argument.
    :param params: An optional list of parameters to be parsed against pre-defined frequently used parameters.
    If set to None (default), command line parameters from sys.argv[1:] are used; setting to an empty list forces
    to use default parameters.
    """
    global common_opts
    global optimizer
    global summary_writer

    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(arg_parser)

    if params is None:
        params = sys.argv[1:]
    common_opts = _get_params(arg_parser, params)

    if common_opts.random_seed is None:
        common_opts.random_seed = random.randint(0, 2**31)
    _set_seed(common_opts.random_seed)

    optimizers = {'adam': torch.optim.Adam,
                 'sgd': torch.optim.SGD,
                 'adagrad': torch.optim.Adagrad}
    if common_opts.optimizer in optimizers:
        optimizer = optimizers[common_opts.optimizer]
    else:
        raise NotImplementedError(f'Unknown optimizer name {common_opts.optimizer}!')

    if summary_writer is None and common_opts.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=common_opts.tensorboard_dir)
        except ModuleNotFoundError:
            print('Cannot load tensorboard module; makes sure you installed everything required')

    return common_opts


def close() -> None:
    """
    Should be called at the end of the program - however, not required unless Tensorboard is used
    """
    global summary_writer
    if summary_writer: summary_writer.close()


def get_opts() -> argparse.Namespace:
    """
    :return: command line options
    """
    global common_opts
    return common_opts


def build_optimizer(params: Iterable) -> torch.optim.Optimizer:
    return optimizer(params, lr=get_opts().lr)


def get_summary_writer() -> 'torch.utils.SummaryWriter':
    """
    :return: Returns an initialized instance of torch.util.SummaryWriter
    """
    global summary_writer
    return summary_writer


def _set_seed(seed) -> None:
    """
    Seeds the RNG in python.random, torch {cpu/cuda}, numpy.
    :param seed: Random seed to be used


    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_sender_receiver(game: torch.nn.Module,
                         dataset: 'torch.utils.data.DataLoader',
                         gs: bool, variable_length: bool,
                         device: Optional[torch.device] = None):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether Gumbel-Softmax relaxation was used during training
    :param variable_length: whether variable-length communication is used
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training  # persist so we restore it back
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, messages, receiver_inputs, receiver_outputs = [], [], [], []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            # Under GS, the only output is a message; under Reinforce, two additional tensors are returned.
            # We don't need them.
            if not gs: message = message[0]

            output = game.receiver(message, receiver_input)
            if not gs: output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                sender_inputs.extend(zip(*sender_input))
            else:
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

            if gs: message = message.argmax(dim=-1)  # actual symbols instead of one-hot encoded

            if not variable_length:
                messages.extend(message)
                receiver_outputs.extend(output)
            else:
                # A trickier part is to handle EOS in the messages. It also might happen that not every message has EOS.
                # We cut messages at EOS if it is present or return the entire message otherwise. Note, EOS id is always
                # set to 0.

                for i in range(message.size(0)):
                    eos_positions = (message[i, :] == 0).nonzero()
                    message_end = eos_positions[0].item() if eos_positions.size(0) > 0 else -1
                    assert message_end == -1 or message[i, message_end] == 0
                    if message_end < 0:
                        messages.append(message[i, :])
                    else:
                        messages.append(message[i, :message_end + 1])

                    if gs:
                        receiver_outputs.append(output[i, message_end, ...])
                    else:
                        receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels


def move_to(x: Any, device: torch.device) \
        -> Any:
    """
    Simple utility function that moves a tensor or a dict/list/tuple of (dict/list/tuples of ...) tensors to a specified device, recursively.
    :param x: tensor, list, tuple, or dict with values that are lists, tuples or dicts with values of ...
    :param device: device to be moved to
    :return: Same as input, but with all tensors placed on device. Non-tensors are not affected. For dicts, the changes are done in-place!
    """
    if hasattr(x, 'to'):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict) or isinstance(x, defaultdict):
        for k, v in x.items():
            x[k] = move_to(v, device)
        return x
    return x


def find_lengths(messages: torch.Tensor) -> torch.Tensor:
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).

    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
    # querying torch.nonzero()
    # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
    # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
    # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
    # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps that happened after 0 occured (including it)
    # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths

