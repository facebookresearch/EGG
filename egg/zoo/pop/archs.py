# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import gumbel_softmax_sample


def get_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "deit": torch.hub.load(
            "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=pretrained
        ),
        "inception": torchvision.models.models.inception_v3(pretrained=pretrained),
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


class SimCLRSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        discrete_evaluation: bool = False,
    ):
        super(SimCLRSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

        self.discrete_evaluation = discrete_evaluation

    def forward(self, resnet_output, sender=False):
        first_projection = self.fc(resnet_output)

        if self.discrete_evaluation and (not self.training) and sender:
            logits = first_projection
            size = logits.size()
            indexes = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits).view(-1, size[-1])
            one_hot.scatter_(1, indexes.view(-1, 1), 1)
            one_hot = one_hot.view(*size)
            first_projection = one_hot

        out = self.fc_out(first_projection)
        return out, first_projection.detach(), resnet_output.detach()


class EmSSLSender(nn.Module):
    def __init__(
        self,
        # input_dim: int # Mat: Add if required to have a no vision version
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        vision_module: Optional[Union[nn.Module, str]] = None,
        pretrained: bool = False,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(EmSSLSender, self).__init__()
        if vision_module is None:
            raise NotImplementedError(
                "Sneder not implemented without vision_module"
            )  # Mat : maybe just remove the optional marker
        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = self.vision_module.fc.in_features
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = get_vision_module(
                vision_module, pretrained=pretrained
            )
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

    def forward(self, input):
        vision_output = self.vision_module(input)
        first_projection = self.fc(vision_output)
        message = gumbel_softmax_sample(
            first_projection, self.temperature, self.training, self.straight_through
        )
        out = self.fc_out(message)
        return out, message.detach(), vision_output.detach()


class Receiver(nn.Module):
    def __init__(
        self,
        # input_dim: int # Mat: Add if required to have a no vision version
        vision_module: Optional[Union[nn.Module, str]] = None,
        pretrained: bool = False,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ):
        super(Receiver, self).__init__()
        if vision_module is None:
            raise NotImplementedError(
                "Receiver not implemented without vision_module"
            )  # Mat : maybe just remove the optional marker
        if isinstance(vision_module, nn.Module):
            self.vision_module = vision_module
            input_dim = self.vision_module.fc.in_features
        elif isinstance(vision_module, str):
            self.vision_module, input_dim = get_vision_module(
                vision_module, pretrained=pretrained
            )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, candidate_images):
        vision_output = self.vision_module(candidate_images)
        return self.fc(vision_output), vision_output.detach()


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)

    def forward(self, sender_input, labels, receiver_input):
        if isinstance(self.sender, SimCLRSender):
            message, message_like, resnet_output_sender = self.sender(
                sender_input, sender=True
            )
            receiver_output, _, resnet_output_recv = self.receiver(receiver_input)
        else:
            message, message_like, resnet_output_sender = self.sender(sender_input)
            receiver_output, resnet_output_recv = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, None
        )

        if hasattr(self.sender, "temperature"):
            if isinstance(self.sender.temperature, torch.nn.Parameter):
                temperature = self.sender.temperature.detach()
            else:
                temperature = torch.Tensor([self.sender.temperature])
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
            aux_input=None,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
