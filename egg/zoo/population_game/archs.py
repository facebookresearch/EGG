# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision

from egg.core.gs_wrappers import gumbel_softmax_sample
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


def get_vision_modules(
    encoder_arch: str, shared: bool = True, pretrain_vision: bool = True
):
    if pretrain_vision:
        assert (
            shared
        ), "A pretrained not shared vision_module is a waste of memory. Please run with --shared set"

    encoder, features_dim = initialize_vision_module(encoder_arch, pretrain_vision)
    encoder_recv = None
    if not shared:
        encoder_recv, _ = initialize_vision_module(encoder_arch)

    return encoder, encoder_recv, features_dim


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None,
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

    def forward(
        self,
        sender,
        receiver,
        loss,
        sender_input,
        labels,
        receiver_input,
        aux_input=None,
    ):
        x_i, x_j = sender_input
        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)

        return self.game(
            sender,
            receiver,
            loss,
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
            aux_input=aux_input,
        )


class VisionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.game = EmComSSLSymbolGame(sender, receiver, loss)
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        x_i, x_j = sender_input
        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
        )


class EmSSLSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(EmSSLSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )
        self.straight_through = straight_through

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, resnet_output, aux_input=None):
        first_projection = self.fc(resnet_output)
        message = gumbel_softmax_sample(
            first_projection, self.temperature, self.training, self.straight_through
        )
        out = self.fc_out(message)
        return out, message.detach(), resnet_output.detach()


class Receiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, resnet_output, aux_input=None):
        return self.fc(resnet_output), resnet_output.detach()


class EmComSSLSymbolGame(nn.Module):
    def __init__(
        self, train_logging_strategy=None, test_logging_strategy=None, *args, **kwargs
    ):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)
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
        receiver_input,
        aux_input=None,
    ):
        message, message_like, resnet_output_sender = sender(sender_input, aux_input)
        receiver_output, resnet_output_recv = receiver(
            message, receiver_input, aux_input
        )

        loss, aux_info = loss(
            _sender_input=sender_input,
            message=message,
            _receiver_input=receiver_input,
            receiver_output=receiver_output,
            _labels=labels,
            _aux_input=aux_input,
        )

        if hasattr(sender, "temperature"):
            if isinstance(sender.temperature, torch.nn.Parameter):
                temperature = sender.temperature.detach()
            else:
                temperature = torch.Tensor([sender.temperature])
            aux_info["temperature"] = temperature

        if not self.training:
            aux_info["message_like"] = message_like
            aux_info["resnet_output_sender"] = resnet_output_sender
            aux_info["resnet_output_recv"] = resnet_output_recv

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction


class UniformAgentSampler(nn.Module):
    # NB: only a module to facilitate checkpoint persistance
    def __init__(self, senders, receivers, losses, seed=1234):
        super().__init__()

        np.random.seed(seed)

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

    def forward(self):
        s_idx, r_idx, l_idx = (
            np.random.choice(len(self.senders)),
            np.random.choice(len(self.receivers)),
            np.random.choice(len(self.losses)),
        )
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


class FullSweepAgentSampler(nn.Module):
    # NB: only a module to facilitate checkpoint persistance
    def __init__(self, senders, receivers, losses):
        super().__init__()

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

        self.senders_order = list(range(len(self.senders)))
        self.receivers_order = list(range(len(self.receivers)))
        self.losses_order = list(range(len(self.losses)))

        self.reset_order()

    def reset_order(self):
        # np.random.shuffle(self.senders_order)
        # np.random.shuffle(self.receivers_order)
        # np.random.shuffle(self.losses_order)

        self.iterator = itertools.product(
            self.senders_order, self.receivers_order, self.losses_order
        )

    def forward(self):
        try:
            sender_idx, recv_idx, loss_idx = next(self.iterator)
        except StopIteration:
            self.reset_order()
            sender_idx, recv_idx, loss_idx = next(self.iterator)
        return (
            self.senders[sender_idx],
            self.receivers[recv_idx],
            self.losses[loss_idx],
            (
                torch.Tensor([sender_idx]).int(),
                torch.Tensor([recv_idx]).int(),
                torch.Tensor([loss_idx]).int(),
            ),
        )


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
