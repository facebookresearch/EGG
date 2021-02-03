# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import egg.core as core


def build_vision_module(model_name: str = "resnet50", vision_hidden_dim: int = 1000):
    vision_module = models.__dict__[model_name]()
    if "resnet" in model_name.lower():
        vision_module.fc = nn.Linear(512, vision_hidden_dim)
    elif "alexnet" in model_name.lower() or model_name.lower() == "vgg11":
        vision_module.classifier[6] = nn.Linear(4096, vision_hidden_dim)
    elif "densenet121" == model_name:
        vision_module.classifier = nn.Linear(1024, vision_hidden_dim)
    else:
        raise NotImplementedError(f"| ERROR: {model_name} vision module is not supported yet")
    return vision_module


def contrastive_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))


def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= 2 * (b - 1)
    return loss


def discriminative_loss(_sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {'acc': acc}


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
        similarity_projection: int = 128
    ):
        super(Receiver, self).__init__()
        self.vision_module = build_vision_module(model_name=model_name, vision_hidden_dim=vision_hidden_dim)
        self.fc_vision = nn.Linear(vision_hidden_dim, similarity_projection)
        self.fc_message = nn.Linear(message_hidden_dim, similarity_projection)

    def forward(self, message, receiver_input=None):
        image_features = self.vision_module()
        embedded_messages = self.fc_message(message)
        embedded_image_features = self.fc_vision(image_features)
        similarity_scores = torch.matmul(embedded_image_features, embedded_messages)
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
        sender,
        receiver,
        discriminative_loss,  # check loss and make it a param
        sender_entropy_coeff=0.1  # TODO, make it a param
    )
    return game
