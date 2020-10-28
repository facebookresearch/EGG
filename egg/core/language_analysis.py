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
    if (t < 0.0).any():
        raise RuntimeError('Encountered negative probabilities')

    t /= t.sum()
    return -(torch.where(t > 0, t.log(), t) * t).sum().item() / np.log(2)


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
    if torch.is_tensor(t) and t.numel() > 1:
        t = tuple(t.tolist())
    elif torch.is_tensor(t) and t.numel() == 1:
        t = t.item()
    return t


def mutual_info(xs, ys):
    """
    I[x, y] = E[x] + E[y] - E[x,y]
    """
    e_x = calc_entropy(xs)
    e_y = calc_entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = calc_entropy(xys)

    return e_x + e_y - e_xy


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
    """
    >>> words = ['cat', 'dog', 'pen', 'ben', 'ten']
    >>> mean = [[ord(c) for c in w] for w in words]
    >>> mess = [[ord(c) for c in w] for w in words]
    >>> round(TopographicSimilarity.compute_topsim(mean, mess, 'hamming', 'hamming'), 6)
    1.0
    >>> round(TopographicSimilarity.compute_topsim(mean, mess, 'hamming', lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2)),  6)
    1.0
    """
    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'hamming',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = False,
                 compute_topsim_test_set: bool = True,
                 is_gumbel: bool = False):

        self.sender_input_distance_fn = sender_input_distance_fn
        self.message_distance_fn = message_distance_fn

        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set
        assert compute_topsim_train_set or compute_topsim_test_set

        self.is_gumbel = is_gumbel

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.print_message(logs, 'train', epoch)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.print_message(logs, 'test', epoch)

    @staticmethod
    def compute_topsim(meanings: torch.Tensor,
                       messages: torch.Tensor,
                       meaning_distance_fn: Union[str, Callable] = 'hamming',
                       message_distance_fn: Union[str, Callable] = 'edit') -> float:

        distances = {'edit': lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
                     'cosine': distance.cosine,
                     'hamming': distance.hamming,
                     'jaccard': distance.jaccard,
                     'euclidean': distance.euclidean,
                     }

        meaning_distance_fn = distances.get(meaning_distance_fn, None) \
            if isinstance(meaning_distance_fn, str) else meaning_distance_fn
        message_distance_fn = distances.get(message_distance_fn, None) \
            if isinstance(message_distance_fn, str) else message_distance_fn

        assert meaning_distance_fn and message_distance_fn, f"Cannot recognize {meaning_distance_fn} or {message_distance_fn} distances"

        meaning_dist = distance.pdist(meanings, meaning_distance_fn)
        message_dist = distance.pdist(messages, message_distance_fn)

        topsim = spearmanr(meaning_dist, message_dist,
                           nan_policy='raise').correlation

        return topsim

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = [msg.tolist() for msg in messages]

        topsim = self.compute_topsim(logs.sender_input, messages)

        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))
        print(output, flush=True)


class PosDisent(Callback):
    """
    Positional disentanglement metric, introduced in "Compositionality and Generalization in Emergent Languages",
    Chaabouni et al., ACL 2020.
    """

    def __init__(self, print_train: bool = False, print_test: bool = True, is_gumbel: bool = False):
        super().__init__()
        assert print_train or print_test, 'At least on of "print_train" and "print_train" must be enabled'

        self.print_train = print_train
        self.print_test = print_test
        self.is_gumbel = is_gumbel

    @staticmethod
    def posdis(attributes, messages):
        """
        Two-symbol messages representing two-attribute world. One symbol encodes on attribute:
        in this case, the metric should be maximized:
        >>> samples = 1_000
        >>> _ = torch.manual_seed(0)
        >>> attribute1 = torch.randint(low=0, high=10, size=(samples, 1))
        >>> attribute2 = torch.randint(low=0, high=10, size=(samples, 1))
        >>> attributes = torch.cat([attribute1, attribute2], dim=1)
        >>> messages = attributes
        >>> PosDisent.posdis(attributes, messages)
        0.9786556959152222
        >>> messages = torch.cat([messages, torch.zeros_like(messages)], dim=1)
        >>> PosDisent.posdis(attributes, messages)
        0.9786556959152222
        """
        gaps = torch.zeros(messages.size(1))
        non_constant_positions = 0.0

        for j in range(messages.size(1)):
            symbol_mi = []
            h_j = None
            for i in range(attributes.size(1)):
                x, y = attributes[:, i], messages[:, j]
                info = mutual_info(x, y)
                symbol_mi.append(info)

                if h_j is None:
                    h_j = calc_entropy(y)

            symbol_mi.sort(reverse=True)

            if h_j > 0.0:
                gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
                non_constant_positions += 1

        score = gaps.sum() / non_constant_positions
        return score.item()

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(
            dim=-1) if self.is_gumbel else logs.message

        posdis = self.posdis(logs.sender_input, message)

        output = json.dumps(dict(posdis=posdis, mode=tag, epoch=epoch))
        print(output, flush=True)

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_message(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        if self.print_test:
            self.print_message(logs, 'test', epoch)
