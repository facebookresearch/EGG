# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn.functional as F


class Loss:
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def __call__(self, sender_input, _message, _receiver_input, receiver_output, labels):
        return self.loss_fn(sender_input, _message, _receiver_input, receiver_output, labels)


class DiscriminationLoss:
    def __call__(self, sender_input, _message, _receiver_input, receiver_output, labels):
        return self.discrimination_loss(receiver_output, labels)

    @staticmethod
    def discrimination_loss(receiver_output, labels):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {'acc': acc}


class RecoLoss:
    def __init__(self, n_attributes: int, n_values: int, batch_size: int):
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.batch_size = batch_size

    def __call__(self, sender_input, _message, _receiver_input, receiver_output, labels):
        self.reconstruction_loss(
            receiver_output,
            labels,
            self.batch_size,
            self.n_attributes,
            self.n_values
        )

    @staticmethod
    def reconstruction_loss(
        receiver_output,
        labels,
        batch_size,
        n_attributes,
        n_values
    ):
        receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
        receiver_guesses = receiver_output.argmax(dim=1)
        correct_samples = (receiver_guesses == labels.view(-1)).view(batch_size, n_attributes).detach()
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        labels = labels.view(batch_size * n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss = loss.view(batch_size, -1).mean(dim=1)
        return loss, {'acc': acc}


class NT_xent_loss:
    def __init__(self, temperature: float = 0.1, similarity: str = "dot"):
        self.temperature = temperature
        self.similarity = similarity

    def __call__(self, _sender_input, message, _receiver_input, receiver_output, _labels):
        self.nt_xent_loss(
            message,
            receiver_output,
            self.similarity,
            self.temperature
        )

    @staticmethod
    def nt_xent_loss(message, receiver_output, similarity, temperature=0.1):
        batch_size, device = message.shape[0], message.device

        n_samples = batch_size * 2
        projs = torch.cat((message, receiver_output))

        if similarity.lower() == "dot":
            logits = projs @ projs.t()
        else:  # add cosine similarity
            raise NotImplementedError(f"| ERROR cannot recognize {similarity} in nt_xent_loss")

        mask = torch.eye(n_samples, device=device).bool()
        logits = logits[~mask].reshape(n_samples, n_samples - 1)
        logits /= temperature

        labels = torch.cat(
            (
                (torch.arange(batch_size, device=device) + batch_size - 1), torch.arange(batch_size, device=device)
            ),
            dim=0
        )
        return F.cross_entropy(logits, labels, reduction='none')
