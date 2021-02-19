# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def get_loss(batch_size: int, temperature: float, device: torch.device):
    nt_xent_entropy = NT_Xent(batch_size, temperature, device)

    def nt_xent_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
        msg, img = receiver_output  # hacky and ugly trick
        loss, acc = nt_xent_entropy(msg, img)
        return loss, {"acc": acc.unsqueeze(0)}

    return nt_xent_loss


class NT_Xent(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss.
    """

    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, msg, img):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017),
        we treat the other 2(N − 1) augmented examples within a minibatch
        as negative examples.

        :param z: Tensor of shape (2 * N, C, H, W) where N is the batch size
            with the first N items are one augmented version of the images (z_i)
            and the last N items are the other augmented version of the same images
            (z_j) in the same order.
            i.e.  z_i, z_j = z[batch_size:, ...], z[:batch_size, ...]
        """
        N = 2 * self.batch_size

        sim = self.similarity_f(msg.unsqueeze(1), img.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        acc = (torch.argmax(logits, dim=1) == labels).float().sum() / N
        return loss, acc
