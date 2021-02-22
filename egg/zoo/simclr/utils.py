# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
from torchvision import datasets


def get_dataloader(
    dataset_name: str,
    dataset_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int
):
    print(f"Using dataset {dataset_name} with image size: {image_size}. ")
    transformation = TransformsIdentity(image_size)
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=transformation
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            dataset_dir,
            train=True,
            download=True,
            transform=transformation
        )
    elif dataset_name == "imagenet":
        train_dataset = datasets.ImageNet(
            dataset_dir,
            "train",
            transform=transformation
        )
    else:
        raise NotImplementedError(f"{dataset_name} is currently not supported.")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
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
