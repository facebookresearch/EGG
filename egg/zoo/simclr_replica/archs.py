# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision

from egg.core.interaction import LoggingStrategy


def get_vision_module(encoder_arch: str):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet18": torchvision.models.resnet18(),
        "resnet34": torchvision.models.resnet34(),
        "resnet50": torchvision.models.resnet50(),
        "resnet101": torchvision.models.resnet101(),
        "resnet152": torchvision.models.resnet152(),
    }
    if encoder_arch not in resnets:
        raise KeyError(f"{encoder_arch} is not a valid ResNet version")

    model = resnets[encoder_arch]
    features_dim = model.fc.in_features
    model.fc = nn.Identity()

    return model, features_dim


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

    def forward(self, sender_input, labels, receiver_input=None, original_image=None):
        message, first_projection, first_projection_bn_relu, resnet_output = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        if (not self.training) and original_image is not None:
            aux_info['original_image'] = original_image
            aux_info["first_projection"] = first_projection.detach()
            aux_info["resnet_output"] = resnet_output.detach()
            aux_info["first_projection_bn_relu"] = first_projection_bn_relu.detach()
        else:
            del original_image
            del first_projection
            del resnet_output
            del first_projection_bn_relu

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output,
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )
        return loss.mean(), interaction


class VisionModule(nn.Module):
    def __init__(self, vision_module: nn.Module):
        super(VisionModule, self).__init__()
        self.encoder = vision_module

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        encoded_input_recv = self.encoder(x_j)
        return encoded_input_sender, encoded_input_recv


class VisionGameWrapper(nn.Module):
    def __init__(self, game: nn.Module, vision_module: nn.Module):
        super(VisionGameWrapper, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None):
        x_i, x_j, original_image = sender_input
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
        visual_features_dim: int,
        output_dim: int
    ):
        super(Sender, self).__init__()
        self.fc = nn.Linear(visual_features_dim, visual_features_dim)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(visual_features_dim, output_dim, bias=False)

    def forward(self, x):
        first_projection = self.fc(x)
        first_projection_with_bn_and_relu = self.bn_relu(first_projection)
        out = self.fc_out(first_projection_with_bn_and_relu)
        return out, first_projection.detach(), first_projection_with_bn_and_relu.detach(), x


class Receiver(nn.Module):
    def __init__(
        self,
        visual_features_dim: int,
        output_dim: int
    ):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_features_dim, visual_features_dim),
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
            nn.Linear(visual_features_dim, output_dim, bias=False)
        )

    def forward(self, x, _input):
        return self.fc(_input)
