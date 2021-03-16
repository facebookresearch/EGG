# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class XEntropyLoss:
    @staticmethod
    def xent_loss(message, receiver_output):
        model_guess = message @ receiver_output
        labels = torch.eye(receiver_output.shape[0])
        acc = (model_guess.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(model_guess, labels, reduction="none")
        return loss, {'acc': acc}

    def __call__(self, _sender_input, message, _receiver_input, receiver_output, _labels):
        return self.loss(message, receiver_output)


class NTXentLoss:
    def __init__(
            self,
            batch_size: int,
            temperature: float = 0.1,
            similarity: str = "cosine"
    ):
        self.temperature = temperature
        self.batch_size = batch_size

        similarities = {"cosine", "dot"}
        assert similarity.lower() in similarities, "Cannot recognize similarity function {similarity}"
        self.similarity = similarity

    def get_similarity_matrix(self, message, receiver_output):
        if self.similarity == "cosine":
            similarity_f = nn.CosineSimilarity(dim=2)
            return similarity_f(message.unsqueeze(1), receiver_output.unsqueeze(0)) / self.temperature
        elif self.similarity == "dot":
            return message @ receiver_output.t()

    def __call__(self, _sender_input, message, _receiver_input, receiver_output, _labels):
        assert message.shape == receiver_output.shape, "Message and receiver output must be of the same size."

        assert not (self.batch_size % 2), f"batch must be multiple of 2, found {self.batch_size} instead"

        similarity_matrix = self.et_similarity_matrix(message, receiver_output)
        sim_msg_img = torch.diag(similarity_matrix, self.batch_size).reshape(self.batch_size, 1)
        sim_img_msg = torch.diag(similarity_matrix, -self.batch_size).reshape(self.batch_size, 1)

        #  TODO: mask might be useless, need to check!
        mask = torch.ones(
            (self.batch_size * 2 , self.batch_size * 2),
            dtype=bool
        ).to(similarity_matrix.device)
        negative_samples = similarity_matrix * mask
        #

        negative_samples_msgs = negative_samples[:self.batch_size, self.batch_size:]
        negative_samples_imgs = negative_samples[self.batch_size:, :self.batch_size]

        labels_msg = torch.zeros(
            self.batch_size
        ).to(
            sim_msg_img.device
        ).long()
        labels_img = torch.zeros_like(labels_msg)

        logits_msg_img = torch.cat(
            (
                sim_msg_img,
                negative_samples_msgs
            ),
            dim=1
        )
        logits_img_msg = torch.cat(
            (
                sim_img_msg,
                negative_samples_imgs
            ),
            dim=1
        )

        loss_msg_img = F.cross_entropy(logits_msg_img, labels_msg, reduction="none")
        loss_img_msg = F.cross_entropy(logits_img_msg, labels_img, reduction="none")
        loss = (loss_msg_img + loss_img_msg) / 2

        model_guesses = torch.argmax(
            torch.cat((logits_msg_img, logits_img_msg), dim=0),
            dim=1
        )
        labels = torch.cat((labels_msg, labels_img), dim=0)
        acc = (model_guesses == labels).float()
        return loss, {"acc": acc}
