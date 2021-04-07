# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision

from egg.core.gs_wrappers import GumbelSoftmaxLayer
from egg.core.interaction import LoggingStrategy


def get_resnet(name, pretrained=False):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in resnets:
        raise KeyError(f"{name} is not a valid ResNet version")

    model = resnets[name]
    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


def get_vision_modules(
    encoder_arch: str,
    shared: bool = False,
    pretrain_vision: bool = False
):
    if pretrain_vision:
        assert shared, "A pretrained not shared vision_module is a waste of memory. Please run with --shared set"

    encoder, features_dim = get_resnet(encoder_arch, pretrain_vision)
    encoder_recv = None
    if not shared:
        encoder_recv, _ = get_resnet(encoder_arch)

    return encoder, encoder_recv, features_dim


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None
    ):
        super(VisionModule, self).__init__()

        self.encoder = sender_vision_module

        self.shared = receiver_vision_module is None
        if not self.shared:
            self.encoder_recv = receiver_vision_module

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        if self.shared:
            encoded_input_recv = self.encoder(x_j)
        else:
            encoded_input_recv = self.encoder_recv(x_j)
        return encoded_input_sender, encoded_input_recv


class VisionGameWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapper, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None):
        original_image = None
        if len(sender_input) == 3:
            x_i, x_j, original_image = sender_input
        else:
            x_i, x_j = sender_input

        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)
        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
            original_image=original_image
        )


class Sender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(Sender, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.gs_layer = GumbelSoftmaxLayer(
            temperature,
            trainable_temperature,
            straight_through
        )
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        fwd_out = self.fwd(x)
        message = self.gs_layer(fwd_out)
        out = self.fc(message)
        return out, message, fwd_out, x


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super(Receiver, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, input):
        return self.fwd(input), input.detach()


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

    def forward(self, sender_input, labels, receiver_input=None, original_image=None):
        message, gs_sample, first_projection, resnet_output = self.sender(sender_input)
        receiver_output, resnet_output_recv = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        aux_info["gs_sample"] = gs_sample.detach()
        if isinstance(self.sender.gs_layer.temperature, torch.nn.Parameter):
            temperature = self.sender.gs_layer.temperature.detach()
        else:
            temperature = torch.Tensor([self.sender.gs_layer.temperature])
        aux_info["temperature"] = temperature

        if (not self.training) and original_image is not None:
            aux_info['original_image'] = original_image
            aux_info['first_projection'] = first_projection.detach()
            aux_info['resnet_output'] = resnet_output.detach()
            aux_info['resnet_output_recv'] = resnet_output_recv.detach()
        else:
            del original_image
            del first_projection
            del resnet_output
            del resnet_output_recv

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
