# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random

import torch
from PIL import ImageFilter
from torchvision import datasets, transforms


def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    is_distributed: bool = False,
    seed: int = 111,
):
    transformations = TransformsAugment(image_size, dataset_name.lower() == "imagenet")
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            dataset_dir, train=True, download=True, transform=transformations
        )
    elif dataset_name == "imagenet":
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    else:
        raise NotImplementedError(f"Cannot recognize dataset {dataset_name}")

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
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
    return train_loader


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TransformsAugment:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, imagenet=True):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transformations = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.ToTensor(),
        ]
        if imagenet:
            transformations.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j
