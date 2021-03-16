# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn as nn
import torchvision


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
        x_i, x_j = sender_input

        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)
        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input
        )


class Sender(nn.Module):
    def __init__(
        self,
        visual_features_dim: int,
        projection_dim: int
    ):
        super(Sender, self).__init__()
        self.fwd = nn.Identity()

        # if projection_dim == -1 we do not apply any nonlinear transformation
        # and simply leave the visual featrues as is
        if projection_dim != -1:
            self.fwd = nn.Sequential(
                nn.Linear(visual_features_dim, projection_dim, bias=False),
                nn.BatchNorm1d(projection_dim),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.fwd(x)


class Receiver(nn.Module):
    def __init__(
        self,
        visual_features_dim: int,
        output_dim: int
    ):
        super(Receiver, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(visual_features_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, _x, input):
        return self.fwd(input)


class SenderGS(nn.Module):
    def __init__(
        self,
        visual_features_dim: int,
        vocab_size: int
    ):
        super(SenderGS, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(visual_features_dim, vocab_size, bias=False),
            nn.BatchNorm1d(vocab_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fwd(x)


class ReceiverGS(nn.Module):
    def __init__(
        self,
        visual_features_dim: int,
        output_dim: int
    ):
        super(ReceiverGS, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(visual_features_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, _x, input):
        return self.fwd(input)
