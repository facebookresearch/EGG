# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cgi import test
from optparse import Option
import random
from typing import Optional

import torch
from PIL import ImageFilter
from sklearn import gaussian_process
from torchvision import datasets, transforms
from torchvision import transforms

def get_test_attack(attck_name:str):
    pass

def collate_fn(batch):
    return (
        torch.stack([x[0][0] for x in batch], dim=0),  # sender_input
        torch.cat([torch.Tensor([x[1]]).long() for x in batch], dim=0),  # labels (original classes, not used in emecom_game)
        torch.stack([x[0][1] for x in batch], dim=0),  # receiver_input
    )

class Gaussian_noise_dataset(torch.utils.data.Dataset):
    def __init__(self, n_images, image_size, n_labels, seed):
        self.n_images = n_images
        self.image_size= image_size
        self.n_labels = n_labels
        self.seed = seed

    def __getitem__(self, idx):
        if idx <= self.n_images:
            gaussian_noise = torch.randn([3, self.image_size, self.image_size],generator = torch.Generator().manual_seed(idx)) # ,torch.Generator().manual_seed(self.seed)
            label = torch.randint(0,self.n_labels,[1],generator = torch.Generator().manual_seed(idx)).item() # random value, not used in com game
            return [gaussian_noise, gaussian_noise], label
        else:
            raise 
    
    def __len__(self):
        return self.n_images

def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 0,
    is_distributed: bool = False,
    use_augmentations: bool = True,
    return_original_image: bool = False,
    seed: int = 111,
    split_set: bool = True,
):
    # Param : split_set : if true will return a training and testing set. Otherwise will load train set only.

    transformations = ImageTransformation(
        image_size, use_augmentations, return_original_image, dataset_name
    )

    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", download=True, transform=transformations
        )
    elif dataset_name == "gaussian_noise":
        # Note : augmentations on gaussian noise make little sense, transformations are ignored
        train_dataset=Gaussian_noise_dataset(
            n_images=204800, # matching cifar100
            image_size=image_size,
            n_labels=100, # matching cifar100, does not matter as random
            seed = seed
            )
    else:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    if split_set:
        test_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset,
            [len(train_dataset) // 10, len(train_dataset) - (len(train_dataset) // 10)],
            torch.Generator().manual_seed(seed),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        return test_loader, train_loader

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
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
        dataset_name: Optional[str] = None,
        test_attack=None,
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
        # elif test_attack is not None:
        #     transformations = [
        #         transforms.Resize(size=(size, size)),
        #         test_attack
        #     ]
        else:
            transformations = [
                transforms.Resize(size=(size, size)),
            ]

        transformations.append(transforms.ToTensor())

        if dataset_name == "imagenet":
            transformations.extend(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.CenterCrop(299),
                ]
            )
        elif dataset_name == "cifar100":
            transformations.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transformations)
        self.test_attack = test_attack is not None
        self.return_original_image = return_original_image
        if self.return_original_image:
            self.original_image_transform = transforms.Compose(
                [transforms.Resize(size=(size, size)), transforms.ToTensor()]
            )

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        if self.return_original_image:
            return x_i, x_j, self.original_image_transform(x)
        if self.test_attack:
            return self.original_image_transform(x), x_j
        return x_i, x_j