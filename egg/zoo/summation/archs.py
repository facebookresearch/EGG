# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F


class Receiver(nn.Module):
    def __init__(self, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, 2)

    def forward(self, x, _input, _aux_input=None):
        return self.output(x)


class Encoder(nn.Module):
    def __init__(self, cell, embed_dim, n_hidden, vocab_size):
        super(Encoder, self).__init__()

        self.encoder_cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNN(
                input_size=embed_dim,
                batch_first=True,
                hidden_size=n_hidden,
                num_layers=1,
            )
        elif cell == "gru":
            self.cell = nn.GRU(
                input_size=embed_dim,
                batch_first=True,
                hidden_size=n_hidden,
                num_layers=1,
            )
        elif cell == "lstm":
            self.cell = nn.LSTM(
                input_size=embed_dim,
                batch_first=True,
                hidden_size=n_hidden,
                num_layers=1,
            )
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x, _aux_input=None):
        messages, lengths = x
        emb = self.embedding(messages)

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        _, rnn_hidden = self.cell(packed)
        if isinstance(rnn_hidden, tuple):  # lstm returns c_n, don't need it
            rnn_hidden = rnn_hidden[0]
        hidden = rnn_hidden.squeeze(0)
        return hidden
