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
    batch_size: int = 32,
    num_workers: int = 4,
    is_distributed: bool = False,
    use_augmentations: bool = True,
    return_original_image: bool = False,
    seed: int = 111,
    image_size: int = 32,
):

    transformations = ImageTransformation(
        image_size, use_augmentations, return_original_image, dataset_name
    )

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transformations
        )
    else:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)

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
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self,
        size: int,
        augmentation: bool = False,
        return_original_image: bool = False,
        dataset_name: str = "imagenet",
    ):
        if augmentation:
            s = 1
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
            ]
        else:
            transformations = [transforms.Resize(size=(size, size))]

        if dataset_name in ["imagenet", "cifar10"]:
            if dataset_name == "imagenet":
                m = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            elif dataset_name == "cifar10":
                m = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]

            transformations.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=m, std=std),
                ]
            )
        else:
            transformations.extend([transforms.ToTensor()])

        self.transform = transforms.Compose(transformations)

        self.return_original_image = return_original_image
        if self.return_original_image:
            self.original_image_transform = transforms.Compose(
                [transforms.Resize(size=(size, size)), transforms.ToTensor()]
            )

    def __call__(self, x):
        x_i = self.transform(x)
        x_j = self.transform(x)
        if self.return_original_image:
            return x_i, x_j, self.original_image_transform(x)
        return x_i, x_j
