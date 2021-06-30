# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

from .interaction import LoggingStrategy


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    training: bool = True,
    straight_through: bool = False,
):

    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample


class GumbelSoftmaxLayer(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(GumbelSoftmaxLayer, self).__init__()
        self.straight_through = straight_through

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

    def forward(self, logits: torch.Tensor):
        return gumbel_softmax_sample(
            logits, self.temperature, self.training, self.straight_through
        )


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for an agent that outputs a single symbol. Assumes that during the forward pass,
    the agent returns log-probabilities over the potential output symbols. During training, the wrapper
    transforms them into a sample from the Gumbel Softmax (GS) distribution;
    eval-time it returns greedy one-hot encoding of the same shape.

    >>> inp = torch.zeros((4, 10)).uniform_()
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2))(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2), straight_through=True)(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> (max_value, _), (min_value, _) = outp.max(dim=-1), outp.min(dim=-1)
    >>> (max_value == 1.0).all().item() == 1 and (min_value == 0.0).all().item() == 1
    True
    """

    def __init__(
        self,
        agent,
        temperature=1.0,
        trainable_temperature=False,
        straight_through=False,
    ):
        """
        :param agent: The agent to be wrapped. agent.forward() has to output log-probabilities over the vocabulary
        :param temperature: The temperature of the Gumbel Softmax distribution
        :param trainable_temperature: If set to True, the temperature becomes a trainable parameter of the model
        :params straight_through: Whether straigh-through Gumbel Softmax is used
        """
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sample(
            logits, self.temperature, self.training, self.straight_through
        )
        return sample


class SymbolGameGS(nn.Module):
    """
    Implements one-symbol Sender/Receiver game. The loss must be differentiable wrt the parameters of the agents.
    Typically, this assumes Gumbel Softmax relaxation of the communication channel.
    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc_out = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))
    ...     def forward(self, x, _aux_input=None):
    ...         return self.fc_out(x)
    >>> sender = Sender()
    >>> class Receiver(nn.Module):
    ...     def forward(self, x, _input=None, _aux_input=None):
    ...         return x
    >>> receiver = Receiver()
    >>> def mse_loss(sender_input, _1, _2, receiver_output, _3, _4):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {}

    >>> game = SymbolGameGS(sender=sender, receiver=Receiver(), loss=mse_loss)
    >>> loss, interaction = game(torch.ones((2, 10)), None) #  the second argument is labels, we don't need any
    >>> interaction.aux
    {}
    >>> (loss > 0).item()
    1
    """

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
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
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.
        """
        super(SymbolGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction


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
        if isinstance(x, torch.LongTensor) or (
            torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)
        ):
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
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

    def forward(self, message, input=None, aux_input=None):
        embedded_message = self.embedding(message)
        return self.agent(embedded_message, input, aux_input)


class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc_out = nn.Linear(10, 5) #  input size 10, the RNN's hidden size is 5
    ...     def forward(self, x, _aux_input=None):
    ...         return self.fc_out(x)
    >>> agent = Sender()
    >>> agent = RnnSenderGS(agent, vocab_size=2, embed_dim=10, hidden_size=5, max_len=3, temperature=1.0, cell='lstm')
    >>> output = agent(torch.ones((1, 10)))
    >>> output.size()  # batch size x max_len+1 x vocab_size
    torch.Size([1, 4, 2])
    """

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        temperature,
        cell="rnn",
        trainable_temperature=False,
        straight_through=False,
    ):
        super(RnnSenderGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.agent(x, aux_input)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []

        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence


class RnnReceiverGS(nn.Module):
    """
    Gumbel Softmax-based wrapper for Receiver agent in variable-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector. Since, due to the relaxation, end-of-sequence symbol might have non-zero probability at
    each timestep of the message, `RnnReceiverGS` is applied for each timestep. The corresponding EOS logic
    is handled by `SenderReceiverRnnGS`.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell="rnn"):
        super(RnnReceiverGS, self).__init__()
        self.agent = agent

        self.cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None, aux_input=None):
        outputs = []

        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            outputs.append(self.agent(h_t, input, aux_input))
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)

        return outputs


class SenderReceiverRnnGS(nn.Module):
    """
    This class implements the Sender/Receiver game mechanics for the Sender/Receiver game with variable-length
    communication messages and Gumber-Softmax relaxation of the channel. The vocabulary term with id `0` is assumed
    to the end-of-sequence symbol. It is assumed that communication is stopped either after all the message is processed
    or when the end-of-sequence symbol is met.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 5)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> sender = Sender()
    >>> sender = RnnSenderGS(sender, vocab_size=2, embed_dim=3, hidden_size=5, max_len=3, temperature=5.0, cell='gru')
    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(7, 10)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> receiver = RnnReceiverGS(Receiver(), vocab_size=2, embed_dim=4, hidden_size=7, cell='rnn')
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, labels, aux_input):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {'aux': torch.zeros(sender_input.size(0))}
    >>> game = SenderReceiverRnnGS(sender, receiver, loss)
    >>> loss, interaction = game(torch.ones((3, 10)), None, None)  # batch of 3 10d vectors
    >>> interaction.aux['aux'].detach()
    tensor([0., 0., 0.])
    >>> loss.item() > 0
    True
    """

    def __init__(
        self,
        sender,
        receiver,
        loss,
        length_cost=0.0,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
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
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.

        """
        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0
        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input,
            )
            eos_mask = message[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = expected_length

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction
