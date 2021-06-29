# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(
    temperature: float = 1.0, similarity: str = "cosine", loss_type: str = "xent"
):
    if loss_type.lower() == "xent":
        return XEntLoss(temperature=temperature, similarity=similarity)
    elif loss_type.lower() == "ntxent":
        return NTXentLoss(temperature=temperature, similarity=similarity)
    else:
        raise NotImplementedError(f"ERROR: cannot recognize {loss_type} loss")


class Loss:
    def __init__(self, temperature: float = 1.0, similarity: str = "cosine"):
        self.temperature = temperature

        similarities = {"cosine", "dot"}
        assert (
            similarity.lower() in similarities
        ), "Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    def get_similarity_matrix(
        self,
        message: torch.Tensor,
        receiver_output: torch.Tensor,
        similarity: str = "cosine",
    ):
        if self.similarity == "cosine":
            similarity_f = nn.CosineSimilarity(dim=2)
            return (
                similarity_f(message.unsqueeze(1), receiver_output.unsqueeze(0))
                / self.temperature
            )
        elif self.similarity == "dot":
            return message @ receiver_output.t()


class XEntLoss(Loss):
    def xent_loss(self, message: torch.Tensor, receiver_output: torch.Tensor):
        batch_size = receiver_output.shape[0]
        model_guesses = self.get_similarity_matrix(message, receiver_output)

        labels = torch.arange(batch_size, device=message.device)
        acc = (model_guesses.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(model_guesses, labels, reduction="none")
        return loss, {"acc": acc, "game_acc": acc}

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        assert (
            message.shape == receiver_output.shape
        ), "Message and receiver output must be of the same size."
        return self.xent_loss(message, receiver_output)


class NTXentLoss(Loss):
    def ntxent_loss(self, message, receiver_output):
        batch_size = message.shape[0]

        input = torch.cat((message, receiver_output), dim=0)

        similarity_matrix = self.get_similarity_matrix(input, input)

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

        acc = (
            (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        )  # this is soft_acc

        logits_msg_img = similarity_matrix[:batch_size, batch_size:]
        labels_msg_img = torch.arange(batch_size, device=message.device)
        game_acc = (
            (torch.argmax(logits_msg_img, dim=1) == labels_msg_img).float().detach()
        )

        return loss, {"acc": acc, "game_acc": game_acc}

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        return self.ntxent_loss(message, receiver_output)
