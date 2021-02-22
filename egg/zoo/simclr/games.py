# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from egg.core.continous_communication import ContinuousLinearSender, SenderReceiverContinuousCommunication
from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.losses import get_loss


def get_resnet(name, pretrained=False):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in resnets:
        raise KeyError(f"{name} is not a valid ResNet version")

    model = resnets[name]
    n_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, n_features


class VisionModule(nn.Module):
    def __init__(
        self,
        encoder_arch: str,
        projection_dim: int,
        shared: bool = False
    ):
        super(VisionModule, self).__init__()

        self.encoder, features_dim = get_resnet(encoder_arch)
        self.fc = nn.Sequential(
            nn.Linear(features_dim, projection_dim, bias=False),
            nn.ReLU(),
        )
        self.shared = shared

        if not shared:
            self.encoder_recv, _ = get_resnet(encoder_arch)
            self.fc_recv = nn.Sequential(
                nn.Linear(features_dim, projection_dim, bias=False),
                nn.ReLU(),
            )

    def forward(self, input):
        encoded_input_sender = self.fc(self.encoder(input))
        encoded_input_recv = encoded_input_sender
        if not self.shared:
            encoded_input_recv = self.fc_recv(self.encoder_recv(input))

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
        x_i, x_j = sender_input
        _sender_input = torch.cat([x_i, x_j], dim=0)

        sender_encoded_input, receiver_encoded_input = self.vision_module(_sender_input)
        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input
        )


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


def build_game(opts):
    device = torch.device("cuda" if opts.cuda else "cpu")
    vision_encoder = VisionModule(
        encoder_arch=opts.model_name,
        projection_dim=opts.vision_projection_dim,
        shared=opts.shared_vision
    )

    loss = get_loss(opts.batch_size, opts.ntxent_tau, device)

    sender = ContinuousLinearSender(
        agent=nn.Identity(),
        encoder_input_size=opts.vision_projection_dim,
        encoder_hidden_size=opts.sender_output_size
    )
    receiver = Receiver(
        msg_input_dim=opts.sender_output_size,
        img_feats_input_dim=opts.vision_projection_dim,
        output_dim=opts.receiver_output_size
    )
    train_logging_strategy = LoggingStrategy.minimal()
    game = SenderReceiverContinuousCommunication(sender, receiver, loss, train_logging_strategy)
    game = VisionGameWrapper(game, vision_encoder)
    return game
