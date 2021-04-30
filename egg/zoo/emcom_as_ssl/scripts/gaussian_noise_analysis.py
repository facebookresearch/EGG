# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# python -m egg.zoo.emcom_as_ssl.scripts.gaussian_noise_analysis \
#    --loss_type="comm_ntxent" \
#    --checkpoint_path="<path_to_checkpoint_folder>/40329422_0/final.tar" \
#    --evaluate_with_augmentations


import argparse
from typing import Callable, Optional

import torch
from torchvision import transforms

from egg.zoo.emcom_as_ssl.scripts.utils import add_common_cli_args, evaluate, get_game, get_params


def get_random_noise_dataloader(
    dataset_size: int = 49152,
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 4,
    use_augmentations: bool = False,
):

    transformations = TransformsGaussianNoise(augmentations=use_augmentations)
    dataset = GaussianNoiseDataset(size=dataset_size, image_size=image_size, transformations=transformations)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            size: int = 3072,
            image_size: int = 224,
            transformations: Optional[Callable] = None
    ):
        self.transformations = transformations
        self.image_size = image_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = torch.randn(3, self.image_size, self.image_size)
        if self.transformations:
            sample = self.transformations(sample)
        return sample, torch.zeros(1)


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
        return x_i, x_j


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        simclr_sender=cli_args.simclr_sender,
        shared_vision=cli_args.shared_vision,
        loss_type=cli_args.loss_type
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    dataloader = get_random_noise_dataloader(use_augmentations=cli_args.evaluate_with_augmentations)

    print("| Starting evaluation ...")
    loss, soft_acc, game_acc, _ = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}")


if __name__ == "__main__":
    main()
