# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
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
    def __init__(
            self,
            batch_size: int,
            temperature: float = 0.1,
            similarity: str = "cosine",
            normalize_before: bool = True
    ):
        self.temperature = temperature
        self.batch_size = batch_size

        similarities = {"cosine", "dot"}
        assert similarity.lower() in similarities, f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity

        self.normalized_before = normalize_before

    def __call__(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        input = receiver_output
        if self.normalize_before:
            input = F.normalize(input, dim=1)

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
