# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
from torchvision import datasets


def get_dataloader(opts):
    print(f"Using dataset {opts.dataset_name} with image size: {opts.image_size}. ")

    if opts.dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            opts.dataset_dir,
            train=True,
            download=True,
            transform=TransformsIdentity(opts.image_size)
        )
    else:
        raise NotImplementedError(f"{opts.dataset_name} is currently not supported.")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    return train_loader


class TransformsIdentity:
    """
    A minimal transformation module that transforms any given data example
    into two identical views of the same example, denoted x ̃i and x ̃j,
    which we consider as a positive pair.
    """

    def __init__(self, size):
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j
