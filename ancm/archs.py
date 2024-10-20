import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional
from collections import defaultdict

from egg.core.reinforce_wrappers import RnnEncoder
from egg.core.baselines import Baseline, MeanBaseline
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths


# GS classes
class SenderGS(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderGS, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverGS(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverGS, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


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
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        max_len: int,
        vocab_size: int,
        erasure_pr: float = 0.0,
        device: torch.device = torch.device("cpu"),
        length_cost=0.0,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
        seed: int = 42,
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
        self.channel = ErasureChannel(erasure_pr, max_len, vocab_size, device, False, seed)
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

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None, apply_noise=True):
        message = self.sender(sender_input, aux_input)

        message = self.channel(message, message_length=None, apply_noise=apply_noise)
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


def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {"accuracy": acc * 100}


# Reinforce
class SenderReinforce(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverReinforce(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        energies = energies.squeeze()
        return self.logsoft(energies)


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """

    def __init__(
        self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1
    ):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size+1, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, aux_input=None, lengths=None):
        encoded = self.encoder(message, lengths)
        sample, logits, entropy = self.agent(encoded, input, aux_input)

        return sample, logits, entropy


def loss_reinforce(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output == _labels).detach().float()
    return -acc, {'accuracy': acc * 100}


class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce
    the variance of the gradient estimate.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(3, 10)
    ...     def forward(self, rnn_output, _input=None, _aux_input=None):
    ...         return self.fc(rnn_output)
    >>> sender = Sender()
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input=None, _aux_input=None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    ...     loss = F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1)
    ...     aux = {'aux': torch.ones(sender_input.size(0))}
    ...     return loss, aux
    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((5, 3)).normal_()
    >>> optimized_loss, interaction = game(input, labels=None, aux_input=None)
    >>> sorted(list(interaction.aux.keys()))  # returns debug info such as entropies of the agents, message length etc
    ['aux', 'length', 'receiver_entropy', 'sender_entropy']
    >>> interaction.aux['aux'], interaction.aux['aux'].sum()
    (tensor([1., 1., 1., 1., 1.]), tensor(5.))
    """

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        max_len: int,
        vocab_size: int,
        erasure_pr: float = 0.0,
        device: torch.device = torch.device("cpu"),
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
        seed: int = 42,
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

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.mechanics = CommunicationRnnReinforce(
            sender_entropy_coeff,
            receiver_entropy_coeff,
            max_len,
            vocab_size,
            erasure_pr,
            device,
            length_cost,
            baseline_type,
            train_logging_strategy,
            test_logging_strategy,
            seed)
        self.channel = self.mechanics.channel

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None, apply_noise=False):
        return self.mechanics(
            self.sender,
            self.receiver,
            self.loss,
            sender_input,
            labels,
            receiver_input,
            aux_input,
            apply_noise=apply_noise,
        )


class CommunicationRnnReinforce(nn.Module):
    def __init__(
        self,
        sender_entropy_coeff: float,
        receiver_entropy_coeff: float,
        max_len: int,
        vocab_size: int,
        erasure_pr: float = 0.0,
        device: torch.device = torch.device('cpu'),
        length_cost: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
        seed: int = 42,
    ):
        """
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
        super().__init__()

        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost
        self.channel = ErasureChannel(erasure_pr, max_len, vocab_size, device, False, seed)

        self.baselines = defaultdict(baseline_type)
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

    def forward(
        self,
        sender,
        receiver,
        loss,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
        apply_noise=True,
    ):
        message, log_prob_s, entropy_s = sender(sender_input, aux_input)
        message_length = find_lengths(message)
        message = self.channel(message, message_length, apply_noise)  # apply noise

        receiver_output, log_prob_r, entropy_r = receiver(
            message, receiver_input, aux_input, message_length
        )

        loss, aux_info = loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_length).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_length.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
            + entropy_r.mean() * self.receiver_entropy_coeff
        )

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_length.float() * self.length_cost

        policy_length_loss = (
            (length_loss - self.baselines["length"].predict(length_loss))
            * effective_log_prob_s
        ).mean()
        policy_loss = (
            (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        ).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.baselines["loss"].update(loss)
            self.baselines["length"].update(length_loss)

        aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()
        aux_info["length"] = message_length.float()  # will be averaged

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )

        return optimized_loss, interaction


# BEC class
class ErasureChannel(nn.Module):
    """
    Erases a symbol from a message with a given probability
    """

    def __init__(self, erasure_pr, max_len, vocab_size, device, is_relative_detach=True, seed=42):
        super().__init__()
        self.p = erasure_pr
        self.max_len = max_len
        self.erased_symbol = vocab_size
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def forward(self, message, message_length=None, apply_noise=False):
        # GS: append 0 vector representing Pr[erased_symbol] for all messages
        if message.dim() == 3:
            message = torch.cat([message, torch.zeros((32, 6, 1))], dim=2)

        if self.p != 0 and apply_noise:

            # Reinforce
            if message.dim() == 2:
                # sample symbol indices to be erased
                erase_idx = (torch.rand(*message.size(), generator=self.generator) < self.p) 

                if message_length is None:  # if message length is not provided, compute it
                    message_length = find_lengths(message)

                # True for all message symbols before the 1st EOS symbol
                not_eosed = (
                    torch.stack(
                        [torch.arange(0, message.size(1), requires_grad=False)]) 
                    < torch.cat(
                        [torch.unsqueeze(message_length-1, dim=-1) for _ in range(message.size(1))],
                        dim=1))

                message = torch.where(
                    torch.logical_and(erase_idx, not_eosed),
                    self.erased_symbol,
                    message)

            # GS
            else:
                # randomply sample symbol indices to erase before EOS
                erase_idx = (torch.rand(*message.size()[:-1], generator=self.generator) < self.p)
                erase_idx.requires_grad_(False)
                not_eosed = (message[:,:,0] != 1.)
   
                combined = torch.logical_and(erase_idx, not_eosed)
                combined = combined[:,:,None]
                combined = combined.expand(*message.size())

                # for erased symbols – where should we put 0/1?
                erased_digits = torch.cat((torch.zeros(1, 1, 12), torch.ones(1, 1, 1)), dim=2)
                erased_digits = erased_digits.expand(1, message.size(1), message.size(2))
                erased_digits = erased_digits.expand(*message.size())

                # erase
                message = torch.where(
                    torch.logical_and(combined, erased_digits == 0),
                    0,
                    message)
                message = torch.where(
                    torch.logical_and(combined, erased_digits == 1),
                    1,
                    message)

        return message
