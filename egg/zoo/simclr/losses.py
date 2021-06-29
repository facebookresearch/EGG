# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    def __init__(
        self, batch_size: int, temperature: float = 0.1, similarity: str = "cosine"
    ):
        self.temperature = temperature
        self.batch_size = batch_size

        similarities = {"cosine", "dot"}
        assert (
            similarity.lower() in similarities
        ), "Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    def __call__(
        self,
        _sender_input,
        message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        input = torch.cat((message, receiver_output), dim=0)

        if self.similarity == "cosine":
            similarity_f = nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / self.temperature
            )
        elif self.similarity == "dot":
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, self.batch_size)
        sim_j_i = torch.diag(similarity_matrix, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )

        mask = torch.ones(
            (self.batch_size * 2, self.batch_size * 2), dtype=bool
        ).fill_diagonal_(0)

        negative_samples = similarity_matrix[mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="none") / 2
        acc = (torch.argmax(logits.detach(), dim=1) == labels).float()

        return loss, {"acc": acc}
