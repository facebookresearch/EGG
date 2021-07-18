# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision

from egg.core.interaction import LoggingStrategy


def initialize_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


class Sender(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: Optional[int],
        vocab_size: int = 2048,
    ):
        super(Sender, self).__init__()

        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Sender")

        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )

    def forward(self, x, aux_input=None):
        return self.fc(self.vision_module(x))


class Receiver(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):

        super(Receiver, self).__init__()

        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Receiver")
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        self.temperature = temperature

    def forward(self, message, distractors, aux_input=None):
        distractors = self.fc(self.vision_module(distractors))

        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )

        return similarity_scores


class AgentSampler(nn.Module):
    """Random sampler at training time, fullsweep sampler at test time."""

    def __init__(self, senders, receivers, losses, seed=1234):
        super().__init__()

        np.random.seed(seed)

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

        self.senders_order = list(range(len(self.senders)))
        self.receivers_order = list(range(len(self.receivers)))
        self.losses_order = list(range(len(self.losses)))

        self.reset_order()

    def reset_order(self):
        self.iterator = itertools.product(
            self.senders_order, self.receivers_order, self.losses_order
        )

    def forward(self):
        if self.training:
            s_idx, r_idx, l_idx = (
                np.random.choice(len(self.senders)),
                np.random.choice(len(self.receivers)),
                np.random.choice(len(self.losses)),
            )
        else:
            try:
                sender_idx, recv_idx, loss_idx = next(self.iterator)
            except StopIteration:
                self.reset_order()
                sender_idx, recv_idx, loss_idx = next(self.iterator)

        return (
            self.senders[s_idx],
            self.receivers[r_idx],
            self.losses[l_idx],
            (
                torch.Tensor([s_idx]).int(),
                torch.Tensor([r_idx]).int(),
                torch.Tensor([l_idx]).int(),
            ),
        )


class Game(nn.Module):
    def __init__(
        self,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(Game, self).__init__()

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

    def forward(
        self,
        sender,
        receiver,
        loss,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
    ):
        message = sender(sender_input, aux_input)
        receiver_output = receiver(message, receiver_input, aux_input)

        loss, aux_info = loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
            aux_input,
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


class PopulationGame(nn.Module):
    def __init__(self, game, agents_loss_sampler):
        super().__init__()

        self.game = game
        self.agents_loss_sampler = agents_loss_sampler

    def forward(self, *args, **kwargs):
        sender, receiver, loss, idxs = self.agents_loss_sampler()
        sender_idx, recv_idx, loss_idx = idxs
        # creating an aux_input
        args = list(args)
        args[-1] = {
            "sender_idx": sender_idx,
            "recv_idx": recv_idx,
            "loss_idx": loss_idx,
        }

        return self.game(sender, receiver, loss, *args, **kwargs)
