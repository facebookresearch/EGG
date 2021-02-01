# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_loader(opts):
    imagenet_folder = pathlib.Path(opts.data)
    train_path, val_path, test_path = imagenet_folder / 'train', imagenet_folder / 'val', imagenet_folder / 'test'

    transformation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    train_dataset = datasets.ImageFolder(
        train_path,
        transforms.Compose(transformation)
    )

    validation_dataset = datasets.ImageFolder(
        val_path,
        transforms.Compose(transformation)
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transforms.Compose(transformation)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.data_workers,
        pin_memory=True,
        drop_last=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.data_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.data_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, validation_loader, test_loader
