# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from egg.core.rnn import RnnEncoder


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

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

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

    def forward(self, x_i, x_j):
        encoded_input_sender = self.fc(self.encoder(x_i))
        if self.shared:
            encoded_input_recv = self.fc(self.encoder(x_j))
        else:
            encoded_input_recv = self.fc_recv(self.encoder_recv(x_j))
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
        return torch.cat((msg, img), dim=0)


class RnnReceiverDeterministicContrastive(nn.Module):
    def __init__(
        self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1
    ):
        super(RnnReceiverDeterministicContrastive, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message, lengths)
        agent_output = self.agent(encoded, input)

        logits = torch.zeros(agent_output.size(0) // 2).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy
