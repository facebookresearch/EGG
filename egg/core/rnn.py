# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
from .util import find_lengths


class RnnEncoder(nn.Module):
    """Feeds a sequence into an RNN (vanilla RNN, GRU, LSTM) cell and returns a vector representation 
    of it, which is found as the last hidden state of the last RNN layer. Assumes that the eos token has the id equal to 0.
    """

    def __init__(self, vocab_size: int, emb_dim: int, n_hidden: int, cell: str = 'rnn', num_layers: int = 1) -> None:
        super(RnnEncoder, self).__init__()

        self.cell = None
        cell = cell.lower()
        if cell == 'rnn':
            self.cell = nn.RNN(input_size=emb_dim, batch_first=True,
                               hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'gru':
            self.cell = nn.GRU(input_size=emb_dim, batch_first=True,
                               hidden_size=n_hidden, num_layers=num_layers)
        elif cell == 'lstm':
            self.cell = nn.LSTM(input_size=emb_dim, batch_first=True,
                                hidden_size=n_hidden, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, message: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feeds a sequence into an RNN cell and returns the last hidden state of the last layer.
        Arguments:
            message {torch.Tensor} -- A sequence to be processed, a torch.Tensor of type Long, dimensions [B, T]
        Keyword Arguments:
            lengths {Optional[torch.Tensor]} -- An optional Long tensor with messages' lengths. (default: {None})
        Returns:
            torch.Tensor -- A float tensor of [B, H]
        """
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.cell(packed)

        if isinstance(self.cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]
