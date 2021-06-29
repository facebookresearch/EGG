# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.interaction import LoggingStrategy


class ContinuousLinearSender(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        encoder_input_size: int,
        encoder_hidden_size: int = 64,
        num_layers: int = 1,
        activation: str = "relu",
    ):
        super(ContinuousLinearSender, self).__init__()

        self.agent = agent
        activations = {
            "relu": F.relu,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu,
            "identity": nn.Identity(),
        }
        self.activation = activations[activation.lower()]

        encoder_hidden_sizes = [encoder_hidden_size] * num_layers
        encoder_layer_dimensions = [(encoder_input_size, encoder_hidden_sizes[0])]

        for i, hidden_size in enumerate(encoder_hidden_sizes[1:]):
            hidden_shape = (self.encoder_hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)

        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )

    def forward(self, x, aux_input=None):
        x = self.agent(x, aux_input)
        for hidden_layer in self.encoder_hidden_layers[:-1]:
            x = self.activation(hidden_layer(x))
        sender_output = self.encoder_hidden_layers[-1](x)
        return sender_output


class ContinuousLinearReceiver(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
    ):

        super(ContinuousLinearReceiver, self).__init__()
        self.agent = agent

    def forward(self, message, input=None, aux_input=None):
        agent_output = self.agent(message, input, aux_input)
        return agent_output


class SenderReceiverContinuousCommunication(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        """
        :param sender: Sender agent. sender.forward() has to output a continouos vector
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
        super(SenderReceiverContinuousCommunication, self).__init__()
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
            receiver_output=receiver_output,
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )
        return loss.mean(), interaction
