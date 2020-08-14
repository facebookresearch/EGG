# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Callable

from collections import defaultdict

import editdistance
from scipy.spatial import distance
from scipy.stats import spearmanr

import numpy as np
import torch
from .callbacks import Callback
from .interaction import Interaction
import json


def entropy_dict(freq_table):
    """
    >>> d = {'a': 1, 'b': 1}
    >>> np.allclose(entropy_dict(d), 1.0)
    True
    >>> d = {'a': 1, 'b': 0}
    >>> np.allclose(entropy_dict(d), 0.0, rtol=1e-5, atol=1e-5)
    True
    """
    t = torch.tensor([v for v in freq_table.values()]).float()
    return torch.distributions.Categorical(probs=t).entropy().item() / np.log(2)


def calc_entropy(messages):
    """
    >>> messages = torch.tensor([[1, 2], [3, 4]])
    >>> np.allclose(calc_entropy(messages), 1.0)
    True
    """
    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    t = tuple(t.tolist())
    return t


class MessageEntropy(Callback):
    def __init__(self, print_train: bool = True, is_gumbel: bool = False):
        super().__init__()
        self.print_train = print_train
        self.is_gumbel = is_gumbel

    def print_message_entropy(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(
            dim=-1) if self.is_gumbel else logs.message
        entropy = calc_entropy(message)

        output = json.dumps(dict(entropy=entropy, mode=tag, epoch=epoch))
        print(output, flush=True)

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_message_entropy(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        self.print_message_entropy(logs, 'test', epoch)


class TopographicSimilarity(Callback):
    distances = {'edit': lambda x, y: editdistance.eval(x, y) / (len(x) + len(y)) / 2,
                 'cosine': distance.cosine,
                 'hamming': distance.hamming,
                 'jaccard': distance.jaccard,
                 'euclidean': distance.euclidean,
                 }

    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'cosine',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = False,
                 compute_topsim_test_set: bool = True):

        self.sender_input_distance_fn = self.distances.get(sender_input_distance_fn, None) \
            if isinstance(sender_input_distance_fn, str) else sender_input_distance_fn
        self.message_distance_fn = self.distances.get(message_distance_fn, None) \
            if isinstance(message_distance_fn, str) else message_distance_fn
        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set

        assert self.sender_input_distance_fn and self.message_distance_fn, f"Cannot recognize {sender_input_distance_fn} or {message_distance_fn} distances"
        assert compute_topsim_train_set or compute_topsim_test_set

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.compute_similarity(
                sender_input=logs.sender_input, messages=logs.message, epoch=epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.compute_similarity(
                sender_input=logs.sender_input, messages=logs.message, epoch=epoch)

    def compute_similarity(self, sender_input: torch.Tensor, messages: torch.Tensor, epoch: int):
        def compute_distance(_list, distance):
            return [distance(el1, el2)
                    for i, el1 in enumerate(_list[:-1])
                    for j, el2 in enumerate(_list[i+1:])
                    ]

        messages = [msg.tolist() for msg in messages]

        input_dist = compute_distance(
            sender_input.numpy(), self.sender_input_distance_fn)
        message_dist = compute_distance(messages, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist,
                           nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim, epoch=epoch))
        print(output_message, flush=True)
