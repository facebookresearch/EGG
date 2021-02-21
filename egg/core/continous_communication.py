# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch.nn as nn
import torch.nn.functional as F

from egg.core import Interaction


class ContinuousLinearSender(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        encoder_input_size: int,
        encoder_hidden_size: int = 64,
        num_layers: int = 1,
        activation: str = "relu"
    ):
        super(ContinuousLinearSender, self).__init__()

        self.agent = agent
        activations = {"relu": F.relu, "tanh": F.tanh, "leaky_relu": F.leaky_relu, "identity": nn.Identity()}
        self.activation = activations[activation.lower()]

        encoder_hidden_sizes = [encoder_hidden_size] * num_layers
        encoder_layer_dimensions = [(encoder_input_size, encoder_hidden_sizes[0])]

        for i, hidden_size in enumerate(self.encoder_hidden_sizes[1:]):
            hidden_shape = (self.encoder_hidden_sizes[i], hidden_size)
            encoder_layer_dimensions.append(hidden_shape)

        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(*dimensions) for dimensions in encoder_layer_dimensions]
        )

    def forward(self, x):
        x = self.agent(x)
        for hidden_layer in self.decoder_hidden_layers[:-1]:
            x = self.activation(hidden_layer(x))
        sender_output = self.decoder_hidden_layers[-1](x)
        return sender_output


class ContinuousLinearReceiver(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
    ):

        super(ContinuousLinearReceiver, self).__init__()
        self.agent = agent

    def forward(self, message, input=None):
        agent_output = self.agent(message, input)
        return agent_output


class SenderReceiverContinuousCommunication(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
    ):
        super(SenderReceiverContinuousCommunication, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        return loss.mean(), Interaction.empty()
