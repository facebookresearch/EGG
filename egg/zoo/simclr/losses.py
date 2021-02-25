# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    def __init__(
            self,
            batch_size: int,
            temperature: float = 0.1,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            similarity: str = "cosine"
    ):
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

        similarities = {"cosine", "dot"}
        assert similarity.lower() in similarities, "Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    def __call__(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        input = F.normalize(receiver_output, dim=1)

        if self.similarity == "cosine":
            similarity_f = nn.CosineSimilarity(dim=2)
            similarity_matrix = similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / self.temperature
        elif self.similarity == "dot":
            similarity_matrix = input @ input.t()

        N = 2 * self.batch_size

        sim_i_j = torch.diag(similarity_matrix, self.batch_size)
        sim_j_i = torch.diag(similarity_matrix, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )

        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0

        negative_samples = similarity_matrix[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss_ij = F.cross_entropy(logits[:self.batch_size], labels[:self.batch_size], reduction="none")
        loss_ji = F.cross_entropy(logits[self.batch_size:], labels[self.batch_size:], reduction="none")
        loss = (loss_ij + loss_ji) / 2
        acc = (torch.argmax(logits, dim=1) == labels).float()
        return loss, {"acc": acc}
