# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
from PIL import Image
import random

import numpy as np
import torch
import torchvision.transforms as transforms


class _BatchIterator:
    def __init__(self, dataset, batch_size=32, distractors=1, max_targets_seen=-1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distractors = distractors
        self.max_targets_seen = max_targets_seen if max_targets_seen > 0 else len(dataset)

        self.targets_seen = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.targets_seen + self.batch_size > self.max_targets_seen:
            self.targets_seen = 0  # implement drop_last
            raise StopIteration()
        lineup_size = self.batch_size * (self.distractors + 1)
        sampled_idxs = random.sample(range(len(self.dataset)), lineup_size)
        targets_position = torch.from_numpy(
            np.random.randint(
                self.distractors + 1,
                size=self.batch_size
            )
        )
        receiver_input = torch.cat(
            [self.dataset[idx] for idx in sampled_idxs]
        )
        receiver_input = receiver_input.view(self.batch_size, self.distractors + 1, 3, 224, 224)

        targets = []
        for idx in range(self.batch_size):
            targets.append(receiver_input[idx, targets_position[idx], :, : , :].unsqueeze(0))

        receiver_input = receiver_input.view(self.batch_size * (self.distractors + 1), 3, 224, 224)

        targets = torch.cat(targets, dim=0)
        self.targets_seen += self.batch_size
        return targets, targets_position, receiver_input

    def _sample_target_position_and_add_it_in_right_position(self, img_lineup):
        target_position = random.randrange(self.distractors + 1)

        tmp_img = img_lineup[target_position]
        img_lineup[target_position] = img_lineup[-1]
        img_lineup[-1] = tmp_img

        return img_lineup, target_position


class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transformations=None):
        super(ImagenetDataset, self).__init__()

        with open(data_path) as fd:
            self.paths = fd.readlines()[:-1]  # removing last newline from file
        self.transformations = transformations

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx].strip()).convert("RGB")
        img.load()
        if self.transformations:
            img = self.transformations(img)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        return img.unsqueeze(0)  # 1 X C X H X W


class ImagenetDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.distractors = kwargs.pop('distractors')
        self.max_targets_seen = kwargs.pop('max_targets_seen')
        super(ImagenetDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        return _BatchIterator(self.dataset, self.batch_size, self.distractors, self.max_targets_seen)


def get_loader(opts):
    train_path = pathlib.Path(opts.data)

    transformations = [
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    train_dataset = ImagenetDataset(
        data_path=train_path,
        transformations=transforms.Compose(transformations)
    )

    train_loader = ImagenetDataLoader(
        dataset=train_dataset,
        distractors=opts.distractors,
        max_targets_seen=opts.max_targets_seen,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    return train_loader
