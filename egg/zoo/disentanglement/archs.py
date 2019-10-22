# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import random

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x



class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_inputs, n_hidden, bias=False)
        self.fc = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        x = self.emb(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        return x


class PositionalSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, max_len):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len

        log = 0
        k = 1

        while k < n_values:
            k *= (vocab_size - 1)
            log += 1

        assert log * n_attributes < max_len

        self.mapping = nn.Embedding(n_values, log)
        torch.nn.init.zeros_(self.mapping.weight)

        for i in range(n_values):
            value = i
            for k in range(log):
                self.mapping.weight[i, k] = 1 + value % (vocab_size - 1) # avoid putting zeros!
                value = value // (vocab_size - 1)

        assert (self.mapping.weight < vocab_size).all()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_attributes, self.n_values)
        x = x.argmax(dim=-1).view(batch_size * self.n_attributes)
        with torch.no_grad():
            x = self.mapping(x)
        x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x, zeros, zeros


class LinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc = nn.Linear(vocab_size * max_length, n_outputs)

    def forward(self, x, *rest):
        assert x.size(1) <= self.max_length, f'{x.size()}, {self.max_length}'
        board = torch.zeros(x.size(0), self.max_length, self.vocab_size, requires_grad=False, device=x.device)

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                board[i, j, x[i, j]] = 1
        board = board.view(board.size(0), -1)

        result = self.fc(board)
        #result = result.unsqueeze(1)
        #result = result.expand(result.size(0), x.size(1), result.size(-1))

        zeros = torch.zeros(x.size(0), device=x.device)
        return result, zeros, zeros


class BosSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, max_len):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len

        assert (n_attributes * n_values) <= max_len
        assert vocab_size - 1 >= n_attributes

        # each attribute uses unique symbol, avoiding zero
        self.attribute2symbol = [x + 1 for x in random.sample(range(vocab_size - 1), n_attributes)]


    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, self.n_attributes, self.n_values)
        x = x.argmax(dim=-1).view(batch_size, self.n_attributes)

        result = torch.zeros(x.size(0), self.max_len, requires_grad=False).long()
        attribs = list(range(self.n_attributes))
        
        for i in range(x.size(0)):
            current_position = 0
            random.shuffle(attribs)

            for j in attribs:
                attr_value = x[i, j]
                result[i, current_position: current_position + attr_value] = self.attribute2symbol[j]
                current_position += attr_value
        result = result.to(x.device)

        """result = []
        attribs = list(range(self.n_attributes))
        
        for i in range(x.size(0)):
            current_position = 0
            random.shuffle(attribs)
            result.append([])

            for j in attribs:
                attr_value = x[i, j]
                result[-1].extend([self.attribute2symbol[j]] * attr_value.item())
            result[-1].extend([0] * (self.max_len - len(result[i])))

        result = torch.tensor(result, device=x.device)
        """
        zeros = torch.zeros(batch_size, result.size(1), device=x.device)
        return result, zeros, zeros


if __name__ == '__main__':
    mapper = BosSender(n_attributes=3, n_values=3, vocab_size=10, max_len=15)
    input = torch.tensor([[0., 0., 1., 0., 1., 0., 1., 0., 0.]])
    print(mapper(input))