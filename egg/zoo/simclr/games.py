# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.losses import get_loss


def get_resnet(name, pretrained=False):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")

    model = resnets[name]
    n_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, n_features


class Sender(nn.Module):
    def __init__(self, projection_dim: int, output_dim: int):
        super(Sender, self).__init__()
        self.fc = nn.Linear(projection_dim, output_dim)

    def forward(self, x):
        x = self.fc(F.leaky_relu(x))
        return x


class Receiver(nn.Module):
    def __init__(
        self,
        msg_input_dim: int,
        img_feats_input_dim: int,
        output_dim: int
    ):
        super(Receiver, self).__init__()
        self.fc_message = nn.Linear(msg_input_dim, output_dim)
        self.fc_img_feats = nn.Linear(img_feats_input_dim, output_dim)

    def forward(self, x, _input):
        msg = self.fc_message(F.leaky_relu(x))
        img = self.fc_img_feats(F.leaky_relu(_input))
        return msg, img


class ContinuousSharedVisionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        vision_module: nn.Module,
        features_dim: int,
        projection_dim: int,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(ContinuousSharedVisionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.vision_module = vision_module

        self.fc = nn.Sequential(
            nn.Linear(features_dim, projection_dim, bias=False),
            nn.ReLU(),
        )

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

        # Hacky code to avoid rewrite a data loader
        x_i, x_j = sender_input
        _sender_input = torch.cat([x_i, x_j], dim=0)
        # SimCLR encoding and projection networks
        sender_encoded_input = self.vision_module(_sender_input)
        sender_encoded_input = self.fc(sender_encoded_input)

        message = self.sender(sender_encoded_input)
        receiver_output = self.receiver(message, sender_encoded_input)

        loss, aux_info = self.loss(
            sender_encoded_input, message, receiver_input, receiver_output, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output,  # .detach(), receiver_output is a tuple, cannot detach
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )
        return loss.mean(), interaction


def build_game(opts):
    device = torch.device("cuda" if opts.cuda else "cpu")
    vision_encoder, num_features = get_resnet(opts.model_name, pretrained=False)

    loss = get_loss(opts.batch_size, opts.ntxent_tau, device)

    sender = Sender(opts.projection_dim, opts.sender_output_size)
    receiver = Receiver(opts.sender_output_size, opts.projection_dim, opts.receiver_output_size)

    game = ContinuousSharedVisionGame(
        sender,
        receiver,
        loss,
        vision_encoder,
        num_features,
        opts.projection_dim,
        LoggingStrategy.minimal(),
        LoggingStrategy.minimal()
    )
    return game
