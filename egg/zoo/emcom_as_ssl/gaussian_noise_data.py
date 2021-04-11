# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Optional

import torch
from torchvision import transforms


def get_random_noise_dataloader(
    dataset_size: int = 49152,
    batch_size: int = 128,
    image_size: int = 32,
    num_workers: int = 4,
    use_augmentations: bool = False,
    is_distributed: bool = False,
    seed: int = 111
):

    transformations = TransformsGaussianNoise(augmentation=use_augmentations)
    dataset = GaussianNoiseDataset(size=dataset_size, image_size=image_size, transformations=transformations)

    sampler = None
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=False,
            drop_last=True,
            seed=seed
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            size: int = 49152,
            image_size: int = 224,
            transformations: Optional[Callable] = None
    ):
        self.data = [torch.randn(3, image_size, image_size) for _ in range(size)]
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transformations:
            sample = self.transformations(sample)
        return sample, torch.Tensor([1])


class TransformsGaussianNoise:
    def __init__(self, augmentations: bool = False):
        transformations = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        if augmentations:
            s = 1
            color_jitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            )
            transformations.extend([
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
            ])

        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j, x_i
