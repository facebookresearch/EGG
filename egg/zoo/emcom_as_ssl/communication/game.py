# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models

import egg.core as core
from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.communication.losses import discriminative_loss


def build_vision_module(model_name: str = "resnet50", vision_hidden_dim: int = 1000):
    try:
        vision_module = models.__dict__[model_name](num_classes=vision_hidden_dim)
    except KeyError:
        raise NotImplementedError(f"| ERROR: {model_name} vision module is not supported yet")
    return vision_module


class Sender(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        vision_hidden_dim: int = 128
    ):
        super(Sender, self).__init__()
        self.vision_module = build_vision_module(model_name=model_name, vision_hidden_dim=vision_hidden_dim)

    def forward(self, input):
        return self.vision_module(input)


class Receiver(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        vision_hidden_dim: int = 128,
        message_hidden_dim: int = 128,
        similarity_projection: int = 128,
        distractors: int = 1
    ):
        super(Receiver, self).__init__()
        self.vision_module = build_vision_module(model_name=model_name, vision_hidden_dim=vision_hidden_dim)
        self.fc_vision = nn.Linear(vision_hidden_dim, similarity_projection)
        self.fc_message = nn.Linear(message_hidden_dim, similarity_projection)

        self.vision_hidden_dim = vision_hidden_dim
        self.distractors = distractors

    def forward(self, message, receiver_input=None):
        image_features = self.vision_module(receiver_input)
        image_features = image_features.view((-1, self.distractors + 1, self.vision_hidden_dim))
        embedded_messages = self.fc_message(message).relu()
        embedded_image_features = self.fc_vision(image_features).relu()
        similarity_scores = torch.matmul(embedded_image_features, embedded_messages.unsqueeze(dim=-1)).squeeze()
        return similarity_scores


def build_game(opts):
    sender = Sender(model_name=opts.arch, vision_hidden_dim=opts.sender_hidden)
    sender = core.RnnSenderReinforce(
        sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        cell=opts.sender_cell,
        max_len=opts.max_len
    )
    receiver = Receiver(
        model_name=opts.arch,
        vision_hidden_dim=opts.vision_hidden_dim_receiver,
        message_hidden_dim=opts.receiver_hidden,
        similarity_projection=opts.similarity_projection
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size,
        opts.receiver_embedding,
        opts.receiver_hidden,
        cell=opts.receiver_cell
    )

    game = core.SenderReceiverRnnReinforce(
        sender=sender,
        receiver=receiver,
        loss=discriminative_loss,  # check loss and make it a param
        sender_entropy_coeff=0.1,  # TODO, make it a param
        train_logging_strategy=LoggingStrategy.minimal(),
        test_logging_strategy=LoggingStrategy.minimal()
    )
    return game
