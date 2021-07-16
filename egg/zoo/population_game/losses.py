# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class XEntLoss:
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

    def xent_loss(self, message: torch.Tensor, receiver_output: torch.Tensor):
        batch_size = receiver_output.shape[0]
        model_guesses = self.get_similarity_matrix(message, receiver_output)

        labels = torch.arange(batch_size, device=message.device)
        acc = (model_guesses.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(model_guesses, labels, reduction="none")
        return loss, {
            "acc": acc,
            "game_acc": acc,
            "receiver_guesses": model_guesses.detach(),
        }

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
