# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformedSender(nn.Module):
    def __init__(
        self,
        game_size,
        feat_size,
        embedding_size,
        hidden_size,
        vocab_size=100,
        temp=1.0,
    ):
        super(InformedSender, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temp = temp

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.conv2 = nn.Conv2d(
            1,
            hidden_size,
            kernel_size=(game_size, 1),
            stride=(game_size, 1),
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False
        )
        self.lin4 = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x, _aux_input=None):
        emb = self.return_embeddings(x)

        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv2(emb)
        h = torch.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = torch.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)
        h = self.lin4(h)
        h = h.mul(1.0 / self.temp)
        # h of size (batch_size, vocab_size)
        logits = F.log_softmax(h, dim=1)

        return logits

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x 1 x embedding_size
            embs.append(h_i)
        # concatenate the embeddings
        h = torch.cat(embs, dim=2)

        return h


class Receiver(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size, vocab_size, reinforce):
        super(Receiver, self).__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        if reinforce:
            self.lin2 = nn.Embedding(vocab_size, embedding_size)
        else:
            self.lin2 = nn.Linear(vocab_size, embedding_size, bias=False)

    def forward(self, signal, x, _aux_input=None):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size()) == 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h
