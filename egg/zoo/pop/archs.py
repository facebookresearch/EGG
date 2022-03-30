# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Optional, Union

import timm
import numpy as np
import torch
import torch.nn as nn
import torchvision

from egg.core.interaction import LoggingStrategy


def initialize_vision_module(name: str = "resnet50", pretrained: bool = False):
    print("initialize module", name)
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "inception": torchvision.models.inception_v3(
            pretrained=pretrained, aux_logits=not pretrained
        ),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
        "vit": timm.create_model("vit_base_patch16_384", pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    if name in ["resnet50", "resnet101", "resnet152"]:
        n_features = model.fc.in_features
        model.fc = nn.Identity()

    elif name == "vgg11":
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()

    elif name == "inception":
        n_features = model.fc.in_features
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Identity()
        model.fc = nn.Identity()

    else:  # vit
        n_features = model.head.in_features
        model.head = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        if name == "inception":
            model.aux_logits = False
        # TODO : verify that this is not mistakenly turned back on
        model = (
            model.eval()
        )  # Mat : --> dropout blocked, as well as all other training dependant behaviors

    return model, n_features, name


class Sender(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: Optional[int],
        name: str = "resnet50",
        vocab_size: int = 2048,
    ):
        super(Sender, self).__init__()
        self.name = name
        self.init_vision_module(vision_module, input_dim)
        self.init_com_layer(input_dim, vocab_size)

    def init_vision_module(self, vision_module, input_dim):
        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = input_dim
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = initialize_vision_module(vision_module)
        else:
            raise RuntimeError("Unknown vision module for the Sender")

    def init_com_layer(self, input_dim, vocab_size):
        self.fc = nn.Sequential(
            nn.Linear(input_dim, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        pass

    def forward(self, x, aux_input=None):
        vision_module_out = self.vision_module(x)
        if not self.training:
            aux_input["resnet_output_sender"] = vision_module_out.detach()
        # elif self.name == "inception":
        #     vision_module_out = vision_module_out.logits

        return self.fc(vision_module_out)


class Receiver(nn.Module):
    def __init__(
        self,
        vision_module: Union[nn.Module, str],
        input_dim: int,
        name: str = "resnet50",
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
    ):

        super(Receiver, self).__init__()

        self.name = name

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
        vision_module_out = self.vision_module(distractors)
        # if self.name == "inception":
        #     vision_module_out = vision_module_out.logits
        distractors = self.fc(vision_module_out)

        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )

        if not self.training:
            aux_input["receiver_message_embedding"] = message.detach()

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
            sender_idx, recv_idx, loss_idx = (
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
            self.senders[sender_idx],
            self.receivers[recv_idx],
            self.losses[loss_idx],
            (
                torch.Tensor([sender_idx]).int(),
                torch.Tensor([recv_idx]).int(),
                torch.Tensor([loss_idx]).int(),
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
        # if not self.training:
        sender.to("cuda")  # Mat !! TODO : change this to common opts device
        receiver.to("cuda")
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
        # if not self.training:
        sender.to("cpu")
        receiver.to("cpu")
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
