# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


class DiscriminationLoss:
    def __call__(self, sender_input, _message, _receiver_input, receiver_output, labels):
        return self.discrimination_loss(receiver_output, labels)

    @staticmethod
    def discrimination_loss(receiver_output, labels):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {'acc': acc}


class ReconstructionLoss:
    def __init__(self, n_attributes: int, n_values: int, batch_size: int):
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.batch_size = batch_size

    def __call__(self, sender_input, _message, _receiver_input, receiver_output, labels):
        return self.reconstruction_loss(
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


class NTXentLoss:
    """NTXentLoss as originally described in https://arxiv.org/abs/2002.05709.

    This loss used in self-supervised learning setups requires the two views of the input datapoint
    to be returned distinctly by Sender and Receiver.

    >>> x_i = torch.eye(128)
    >>> x_j = torch.eye(128)
    >>> loss_fn = NTXentLoss(batch_size=128)
    >>> loss, acc_dict = loss(None, a, None, b, None)
    """

    def __init__(
            self,
            temperature: float = 1.0,
            similarity: str = "cosine",
    ):
        self.temperature = temperature

        similarities = {"cosine", "dot"}
        assert similarity.lower() in similarities, f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    @staticmethod
    def ntxent_loss(
        sender_output: torch.Tensor,
        receiver_output: torch.Tensor,
        temperature: float = 1.0,
        similarity: str = "cosine"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        assert sender_output.shape[0] == receiver_output.shape[0]
        batch_size = sender_output.shape[0]

        input = torch.cat((receiver_output, receiver_output), dim=0)

        if similarity == "cosine":
            similarity_matrix = torch.nn.functional.cosine_similarity(
                input.unsqueeze(1),
                input.unsqueeze(0),
                dim=2
            ) / temperature
        elif similarity == "dot":
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_samples = torch.cat(
            (sim_i_j, sim_j_i),
            dim=0
        ).reshape(batch_size * 2, 1)

        mask = torch.ones(
            (batch_size * 2, batch_size * 2),
            dtype=bool
        ).fill_diagonal_(0)

        negative_samples = similarity_matrix[mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="none") / 2

        acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        return loss, {"acc": acc}

    def __call__(self, _sender_input, message, _receiver_input, receiver_output, _labels):
        return self.ntxent_loss(
            message,
            receiver_output,
            temperature=self.temperature,
            similarity=self.similarity
        )
