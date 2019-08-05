# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for an agent that outputs a single symbol. Assumes that during the forward pass,
    the agent returns log-probabilities over the potential output symbols. During training, the wrapper
    transforms them into a sample from the Gumbel Softmax (GS) distribution; eval-time it returns greedy one-hot encoding
    of the same shape.

    The temperature of the GS distribution can be annealed using `update_temp`.
    """
    def __init__(self, agent, temperature=1.0, trainable_temperature=False):
        """
        :param agent: The agent to be wrapped. agent.forward() has to output log-probabilities over the vocabulary
        :param temperature: The temperature of the Gumbel Softmax distribution
        :param trainable_temperature: If set to True, the temperature becomes a trainable parameter of the model
        """
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(torch.tensor([temperature]), requires_grad=True)

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        if self.training:
            return RelaxedOneHotCategorical(logits=logits, temperature=self.temperature).rsample()
        else:
            return torch.zeros_like(logits).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)


class SymbolGameGS(nn.Module):
    """
    Implements one-symbol Sender/Receiver game. The loss must be differentiable wrt the parameters of the agents.
    Typically, this assumes Gumbel Softmax relaxation of the communication channel.
    >>> class Receiver(nn.Module):
    ...     def forward(self, x, _input=None):
    ...         return x

    >>> receiver = Receiver()
    >>> sender = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))

    >>> def mse_loss(sender_input, _1, _2, receiver_output, _3):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {}

    >>> game = SymbolGameGS(sender=sender, receiver=Receiver(), loss=mse_loss)
    >>> forward_result = game(torch.ones((2, 10)), None) #  the second argument is labels, we don't need any
    >>> forward_result[1]
    {}
    >>> (forward_result[0] > 0).item()
    1
    """
    def __init__(self, sender, receiver, loss):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        """
        super(SymbolGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        return loss.mean(), rest_info


class RelaxedEmbedding(nn.Embedding):
    """
    A drop-in replacement for `nn.Embedding` such that it can be used _both_ with Reinforce-based training
    and with Gumbel-Softmax one.
    Important: nn.Linear and nn.Embedding have different initialization strategies, hence replacing nn.Linear with
    `RelaxedEmbedding` might change results.

    >>> emb = RelaxedEmbedding(15, 10)  # vocab size 15, embedding dim 10
    >>> long_query = torch.tensor([[1], [2], [3]]).long()
    >>> long_query.size()
    torch.Size([3, 1])
    >>> emb(long_query).size()
    torch.Size([3, 1, 10])
    >>> float_query = torch.zeros((3, 15)).scatter_(-1, long_query, 1.0).float().unsqueeze(1)
    >>> float_query.size()
    torch.Size([3, 1, 15])
    >>> emb(float_query).size()
    torch.Size([3, 1, 10])

    # make sure it's the same query, one-hot and symbol-id encoded
    >>> (float_query.argmax(dim=-1) == long_query).all().item()
    1
    >>> (emb(float_query) == emb(long_query)).all().item()
    1
    """
    def forward(self, x):
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            return F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(x, self.weight)


class SymbolReceiverWrapper(nn.Module):
    """
    An optional wrapper for single-symbol Receiver, both Gumbel-Softmax and Reinforce. Receives a message, embeds it,
    and passes to the wrapped agent.
    """
    def __init__(self, agent, vocab_size, agent_input_size):
        super(SymbolReceiverWrapper, self).__init__()
        self.agent = agent
        self.embedding = RelaxedEmbedding(vocab_size, agent_input_size)

    def forward(self, message, input=None):
        embedded_message = self.embedding(message)
        return self.agent(embedded_message, input)


class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.

    >>> agent = nn.Linear(10, 5) #  input size 10, the RNN's hidden size is 5
    >>> agent = RnnSenderGS(agent, vocab_size=2, embed_dim=10, hidden_size=5, max_len=3, temperature=1.0, cell='lstm')
    >>> output = agent(torch.ones((1, 10)))
    >>> output.size()  # batch size x max_len x vocab_size
    torch.Size([1, 3, 2])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, temperature, cell='rnn', force_eos=True,
                 trainable_temperature=False):
        super(RnnSenderGS, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if self.force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(torch.tensor([temperature]), requires_grad=True)

        self.cell = None

        cell = cell.lower()

        if cell == 'rnn':
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == 'gru':
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == 'lstm':
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []

        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = RelaxedOneHotCategorical(logits=step_logits, temperature=self.temperature)

            if self.training:
                x = distr.rsample()
            else:
                x = torch.zeros_like(step_logits).scatter_(-1, step_logits.argmax(dim=-1, keepdim=True), 1.0)

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        if self.force_eos:
            eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
            eos[:, 0, 0] = 1
            sequence = torch.cat([sequence, eos], dim=1)

        return sequence


class RnnReceiverGS(nn.Module):
    """
    Gumbel Softmax-based wrapper for Receiver agent in variable-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector. Since, due to the relaxation, end-of-sequence symbol might have non-zero probability at
    each timestep of the message, `RnnReceiverGS` is applied for each timestep. The corresponding EOS logic is handled by
    `SenderReceiverRnnGS`.
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn'):
        super(RnnReceiverGS, self).__init__()
        self.agent = agent

        self.cell = None
        cell = cell.lower()
        if cell == 'rnn':
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == 'gru':
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == 'lstm':
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None):
        outputs = []

        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c)) if prev_hidden is not None else \
                    self.cell(e_t)
            else:
                h_t = self.cell(e_t, prev_hidden)

            outputs.append(self.agent(h_t, input))
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)

        return outputs


class SenderReceiverRnnGS(nn.Module):
    """
    This class implements the Sender/Receiver game mechanics for the Sender/Receiver game with variable-length
    communication messages and Gumber-Softmax relaxation of the channel. The vocabulary term with id `0` is assumed
    to the end-of-sequence symbol. It is assumed that communication is stopped either after all the message is processed
    or when the end-of-sequence symbol is met.

    >>> sender = nn.Linear(10, 5)
    >>> sender = RnnSenderGS(sender, vocab_size=2, embed_dim=3, hidden_size=5, max_len=3, temperature=5.0, cell='gru')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(7, 10)
    ...     def forward(self, x, _input):
    ...         return self.fc(x)
    >>> receiver = RnnReceiverGS(Receiver(), vocab_size=2, embed_dim=4, hidden_size=7, cell='rnn')

    >>> def loss(sender_input, _message, _receiver_input, receiver_output, labels):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {'aux' : 0}
    >>> game = SenderReceiverRnnGS(sender, receiver, loss)
    >>> output = game.forward(torch.ones((3, 10)), None, None)  # batch of 3 10d vectors
    >>> output[1]['aux'].item()
    0.0
    >>> output[0].item() > 0
    True
    """
    def __init__(self, sender, receiver, loss, length_cost=0.0):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(receiver_output.device)
        expected_length = 0.0

        rest = {}
        z = 0.0
        for step in range(receiver_output.size(1)):
            step_loss, step_rest = self.loss(sender_input, message[:, step, ...], receiver_input, receiver_output[:, step, ...], labels)
            eos_mask = message[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_rest.items():
                rest[name] = value * add_mask + rest.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += step_loss * not_eosed_before + self.length_cost * (step + 1.0) * not_eosed_before
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(torch.ones_like(z)), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_rest.items():
            rest[name] = value * not_eosed_before + rest.get(name, 0.0)
        for name, value in rest.items():
            rest[name] = value.mean()

        rest['mean_length'] = expected_length.mean()
        return loss.mean(), rest
