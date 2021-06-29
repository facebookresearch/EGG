# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torchvision


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

    def forward(self, sender_input, labels, receiver_input=None, _aux_input=None):
        x_i, x_j = sender_input
        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
        )


class Sender(nn.Module):
    def __init__(self, visual_features_dim: int, output_dim: int):
        super(Sender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_features_dim, visual_features_dim),
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
            nn.Linear(visual_features_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.fc(x)


class Receiver(nn.Module):
    def __init__(self, visual_features_dim: int, output_dim: int):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_features_dim, visual_features_dim),
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
            nn.Linear(visual_features_dim, output_dim, bias=False),
        )

    def forward(self, x, _input):
        return self.fc(_input)
