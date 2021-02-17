# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision


class TransformsAugment:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j

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
