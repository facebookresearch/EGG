# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
from PIL import ImageFilter
from torchvision import datasets, transforms


def get_dataloader(
    dataset_name: str,
    dataset_dir: str,
    image_size: int = 32,
    batch_size: int = 32,
    validation_dataset_dir: str = None,
    num_workers: int = 4,
    use_augmentations: bool = True,
    imagenet_normalization: bool = True,
    is_distributed: bool = False,
    seed: int = 111
):
    transformations = TransformsIdentity(image_size, imagenet_normalization)
    if use_augmentations:
        transformations = TransformsAugment(image_size, imagenet_normalization)

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=transformations
        )
        if validation_dataset_dir:
            validation_dataset = datasets.CIFAR10(
                validation_dataset_dir,
                train=False,
                download=True,
                transform=transformations
            )

    elif dataset_name == "imagenet":
        train_dataset = datasets.ImageFolder(
            dataset_dir,
            transform=transformations
        )
        if validation_dataset_dir:
            validation_dataset = datasets.ImageFolder(
                validation_dataset_dir,
                transform=transformations
            )
    else:
        raise NotImplementedError(f"{dataset_name} is currently not supported.")

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
            seed=seed
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = None
    if validation_dataset_dir:
        validation_sampler = None
        if is_distributed:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_dataset,
                shuffle=False,
                drop_last=True,
                seed=seed
            )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=(validation_sampler is None),
            sampler=validation_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return train_loader, validation_loader


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TransformsIdentity:
    """
    A minimal transformation module that transforms any given data example
    into two identical views of the same example, denoted x ̃i and x ̃j,
    which we consider as a positive pair.
    """

    def __init__(self, size, imagenet=False):
        transformations = [transforms.Resize(size=(size, size)), transforms.ToTensor()]
        if imagenet:
            transformations.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j


class TransformsAugment:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, imagenet=False):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        transformations = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.ToTensor(),
        ]
        if imagenet:
            transformations.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j
