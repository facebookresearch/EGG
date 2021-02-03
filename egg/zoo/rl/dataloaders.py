# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
from PIL import Image
import random

import torch
import torchvision.transforms as transforms


class _BatchIterator:
    def __init__(self, dataset, batch_size=32, distractors=1, max_targets_seen=-1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distractors = distractors
        self.max_targets_seen = max_targets_seen if max_targets_seen else len(dataset)

        self.targets_seen = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.targets_seen + self.batch_size > self.max_targets_seen:
            raise StopIteration()  # TODO add drop_last
        targets = random.sample(range(len(self.dataset)), self.batch_size)
        target_positions = []
        receiver_input = []
        for target in targets:
            img_lineup = []
            not_found = True
            while not_found:
                img_lineup = random.sample(range(len(self.dataset)), self.distractors)
                if target not in img_lineup:
                    not_found = False
            img_lineup.append(target)
            final_img_lineup, target_position = self._sample_target_position_and_add_it_in_right_position(
                img_lineup
            )
            target_positions.append(target_position)
            t = torch.cat([self.dataset[idx] for idx in final_img_lineup], dim=0)
            receiver_input.append(t.unsqueeze(dim=0))

        targets = torch.cat([self.dataset[idx] for idx in targets])
        target_positions = torch.LongTensor(target_positions)
        receiver_input = torch.cat(receiver_input, dim=0)

        self.targets_seen += self.batch_size
        return targets, target_positions, receiver_input

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
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
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
