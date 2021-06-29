# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


class DiscriminationLoss:
    def __call__(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        return self.discrimination_loss(receiver_output, labels)

    @staticmethod
    def discrimination_loss(receiver_output, labels):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}


class ReconstructionLoss:
    def __init__(self, n_attributes: int, n_values: int, batch_size: int):
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.batch_size = batch_size

    def __call__(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        return self.reconstruction_loss(
            receiver_output, labels, self.batch_size, self.n_attributes, self.n_values
        )

    @staticmethod
    def reconstruction_loss(
        receiver_output, labels, batch_size, n_attributes, n_values
    ):
        receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
        receiver_guesses = receiver_output.argmax(dim=1)
        correct_samples = (
            (receiver_guesses == labels.view(-1))
            .view(batch_size, n_attributes)
            .detach()
        )
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        labels = labels.view(batch_size * n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        loss = loss.view(batch_size, -1).mean(dim=1)
        return loss, {"acc": acc}


class NTXentLoss:
    """NTXentLoss as originally described in https://arxiv.org/abs/2002.05709.

    This loss is used in self-supervised learning setups and requires the two views of the input datapoint
    to be returned distinctly by Sender and Receiver.
    Note that this loss considers in-batch negatives and and negatives samples are taken within each agent
    datapoints i.e. each non-target element in sender_input and in receiver_input is considered a negative sample.

    >>> x_i = torch.eye(128)
    >>> x_j = torch.eye(128)
    >>> loss_fn = NTXentLoss()
    >>> loss, aux = loss_fn(None, x_i, None, x_j, None, None)
    >>> aux["acc"].mean().item()
    1.0
    >>> aux["acc"].shape
    torch.Size([256])
    >>> x_i = torch.eye(256)
    >>> x_j = torch.eye(128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    Traceback (most recent call last):
        ...
    RuntimeError: sender_output and receiver_output must be of the same shape, found ... instead
    >>> _ = torch.manual_seed(111)
    >>> x_i = torch.rand(128, 128)
    >>> x_j = torch.rand(128, 128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    >>> aux['acc'].mean().item() * 100  # chance level with a batch size of 128, 1/128 * 100 = 0.78125
    0.78125
    """

    def __init__(
        self,
        temperature: float = 1.0,
        similarity: str = "cosine",
    ):
        self.temperature = temperature

        similarities = {"cosine", "dot"}
        assert (
            similarity.lower() in similarities
        ), f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    @staticmethod
    def ntxent_loss(
        sender_output: torch.Tensor,
        receiver_output: torch.Tensor,
        temperature: float = 1.0,
        similarity: str = "cosine",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if sender_output.shape != receiver_output.shape:
            raise RuntimeError(
                f"sender_output and receiver_output must be of the same shape, "
                f"found {sender_output.shape} and {receiver_output.shape} instead"
            )
        batch_size = sender_output.shape[0]

        input = torch.cat((sender_output, receiver_output), dim=0)

        if similarity == "cosine":
            similarity_f = torch.nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / temperature
            )
        elif similarity == "dot":
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )

        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).fill_diagonal_(
            0
        )

        negative_samples = similarity_matrix[mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="none") / 2

        acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        return loss, {"acc": acc}

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        return self.ntxent_loss(
            message,
            receiver_output,
            temperature=self.temperature,
            similarity=self.similarity,
        )
