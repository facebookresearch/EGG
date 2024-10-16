import itertools
import os
import pathlib
from functools import reduce

import numpy as np
import torch
from torch.utils import data

from egg.zoo.objects_game.util import compute_binomial


class VectorsLoader:
    def __init__(self, batch_size=32, shuffle_train_data=True, load_data_path=None, seed=42):
        self.perceptual_dimensions = None
        self._n_features = None
        self.n_distractors = None
        self.batch_size = batch_size
        self.train_samples = None
        self.validation_samples = None
        self.test_samples = None
        self.seed = seed
        self.shuffle_train_data = shuffle_train_data
        self.load_data_path = load_data_path

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    def upd_cl_options(self, opts):
        opts.perceptual_dimensions = self.perceptual_dimensions
        opts.n_distractors = self.n_distractors

    def load_data(self, data_file):
        data = np.load(data_file)
        train, train_labels = data["train"], data["train_labels"]
        valid, valid_labels = data["valid"], data["valid_labels"]
        test, test_labels = data["test"], data["test_labels"]

        # train valid and test are of shape b_size X n_distractors+1 X n_features
        self.train_samples = train.shape[0]
        self.validation_samples = valid.shape[0]
        self.test_samples = test.shape[0]

        self.n_distractors = train.shape[1] - 1
        self.perceptual_dimensions = [-1] * train.shape[-1]
        self._n_features = len(self.perceptual_dimensions)

        return (train, train_labels), (valid, valid_labels), (test, test_labels)

    def collate(self, batch):
        tuples, target_idxs = [elem[0] for elem in batch], [elem[1] for elem in batch]
        receiver_input = np.reshape(
            tuples, (self.batch_size, self.n_distractors + 1, -1)
        )
        labels = np.array(target_idxs)
        targets = receiver_input[np.arange(self.batch_size), labels]
        return (
            torch.from_numpy(targets).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(receiver_input).float(),
        )

    def get_iterators(self):
        assert (self.load_data_path), "No data provided"
        train, valid, test = self.load_data(self.load_data_path)

        assert (
            self.train_samples >= self.batch_size
            and self.validation_samples >= self.batch_size
            and self.test_samples >= self.batch_size
        ), "Batch size cannot be smaller than any split size"

        train_dataset = TupleDataset(*train)
        valid_dataset = TupleDataset(*valid)
        test_dataset = TupleDataset(*test)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_it = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
            shuffle=self.shuffle_train_data,
            worker_init_fn=seed_worker,
            generator=g)
        validation_it = data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g)
        test_it = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g)

        return train_it, validation_it, test_it


class TupleDataset(data.Dataset):
    def __init__(self, tuples, target_idxs):
        self.list_of_tuples = tuples
        self.target_idxs = target_idxs

    def __len__(self):
        return len(self.list_of_tuples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.list_of_tuples):
            raise RuntimeError(
                "Accessing dataset through wrong index: < 0 or >= max_len"
            )
        return self.list_of_tuples[idx], self.target_idxs[idx]
