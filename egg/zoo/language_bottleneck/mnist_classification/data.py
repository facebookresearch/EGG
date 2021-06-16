# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class DoubleMnist:
    def __init__(self, loader, label_mapping):
        self.loader = loader
        self.label_mapping = label_mapping

    def __iter__(self):
        return _DoubleIterator(self.loader, self.label_mapping)


class _DoubleIterator:
    def __init__(self, loader, label_mapping):
        self.iter = loader.__iter__()
        self.label_mapping = label_mapping

    def __next__(self):
        batch, labels = list(self.iter.__next__())
        bsz = batch.size(0)

        half1, half2 = batch[: bsz // 2, ...], batch[bsz // 2 :, ...]
        half_label1, half_label2 = labels[: bsz // 2], labels[bsz // 2 :]

        new_half1 = torch.cat([half1, half2], dim=-1)
        new_half1_labels = half_label1 * 10 + half_label2

        new_half2 = torch.cat([half2, half1], dim=-1)
        new_half2_labels = half_label2 * 10 + half_label1

        new_batch = torch.cat([new_half1, new_half2], dim=0)
        new_labels = torch.cat([new_half1_labels, new_half2_labels], dim=0)
        new_labels = self.label_mapping[new_labels]

        assert bsz == new_batch.size(0) and bsz == new_labels.size(0)
        return new_batch, new_labels


class _DoubleMNIST:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, k):
        batch, labels = self.dataset[k]
        bsz = batch.size(0)

        half1, half2 = batch[: bsz // 2], batch[bsz // 2 :]
        half_label1, half_label2 = labels[: bsz // 2], labels[bsz // 2 :]

        new_half1 = torch.cat([half1, half2], dim=-1)
        new_half1_labels = half_label1 * 10 + half_label2

        new_half2 = torch.cat([half2, half1], dim=-1)
        new_half2_labels = half_label2 * 10 + half_label1

        new_batch = torch.cat([new_half1, new_half2], dim=0)
        new_labels = torch.cat([new_half1_labels, new_half2_labels], dim=0)

        return new_batch, new_labels


class TakeFirstLoader:
    def __init__(self, loader, n):

        batches = []
        for i, batch in enumerate(loader):
            if i >= n:
                break
            batches.append(batch)

        self.sender_inputs = torch.cat([x[0] for x in batches])
        self.labels = torch.cat([x[1] for x in batches])

    def __iter__(self):
        return iter([(self.sender_inputs, self.labels)])


class _SplitIterator:
    def __init__(self, loader, rows_sender, rows_receiver, binarize, receiver_bottom):
        self.iter = loader.__iter__()
        self.rows_sender = rows_sender
        self.rows_receiver = rows_receiver
        self.binarize = binarize
        self.receiver_bottom = receiver_bottom

    def __next__(self):
        batch = list(self.iter.__next__())
        if self.binarize:
            batch[0] = (batch[0] > 0.5).float()

        input_sender = batch[0].clone()
        input_receiver = batch[0].clone()
        d = batch[0].size(2)

        if self.receiver_bottom:
            input_sender[:, :, self.rows_sender :, :] = 0.0
            input_receiver[:, :, : d - self.rows_receiver, :] = 0.0
        else:
            input_sender[:, :, : d - self.rows_sender, :] = 0.0
            input_receiver[:, :, self.rows_receiver :, :] = 0.0

        labels = batch[1]

        return input_sender, labels, input_receiver


class SplitImages:
    def __init__(
        self, loader, rows_sender, rows_receiver, binarize=False, receiver_bottom=True
    ):
        self.loader = loader
        self.rows_sender = rows_sender
        self.rows_receiver = rows_receiver
        self.binarize = binarize
        self.receiver_bottom = receiver_bottom

    def __iter__(self):
        return _SplitIterator(
            self.loader,
            self.rows_sender,
            self.rows_receiver,
            self.binarize,
            receiver_bottom=self.receiver_bottom,
        )
