# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import egg.core as core
from egg.core.interaction import LoggingStrategy


class ContinuousGame(nn.Module):
    """
    Dummy game with continuous setting to benefit
    from EGG toolkit training pipeline for our experiments.
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
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters:
            message and receiver_input.
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
        super(ContinuousGame, self).__init__()
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

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, sender_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction

class SimCLRGameWrapper(nn.Module):
    """
    SimCLR game wrapper.

    This wrapper holds the SimCLR model used to encode
    input representations that is shared between
    the sender and the receiver in our game setting.
    """
    def __init__(
            self,
            game: nn.Module,
            encoder: nn.Module,
            features_dim: int,
            projection_dim: int
        ):
        super(SimCLRGameWrapper, self).__init__()
        self.game = game
        # Encoder network
        self.encoder = encoder
        # SimCLR projection head: We use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(features_dim, features_dim, bias=False),
            nn.ReLU(),
            nn.Linear(features_dim, projection_dim, bias=False),
        )

    def forward(self, sender_input, labels, receiver_input=None):
        # Hacky code to avoid rewrite a data loader
        x_i, x_j = sender_input
        _sender_input = torch.cat([x_i, x_j], dim=0)
        # SimCLR encoding and projection networks
        encoded_input = self.encoder(_sender_input)
        encoded_input = self.projector(encoded_input)
        # Forward encoded input as input for the wrapped game
        return self.game.forward(encoded_input, labels, receiver_input)
