# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from torchvision import transforms

from egg.zoo.population_game.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
)


def get_random_noise_dataloader(
    dataset_size: int = 49152,
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 4,
):

    dataset = GaussianNoiseDataset(size=dataset_size, image_size=image_size)

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
        size: int = 49152,
        image_size: int = 224,
    ):
        self.image_size = image_size
        self.size = size
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = torch.randn(3, self.image_size, self.image_size)
        return self.transform(sample), 0, self.transform(sample)


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        n_senders=cli_args.n_senders,
        n_recvs=cli_args.n_recvs,
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    data = get_random_noise_dataloader(batch_size=cli_args.batch_size)

    print("| Starting evaluation ...")
    loss, interaction = evaluate(
        game,
        data,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cli_args.n_senders,
        cli_args.n_recvs,
    )
    print(
        f"| Loss: {loss}, accuracy across all agents (out of 100): {interaction.aux['acc'].mean().item() * 100}"
    )


if __name__ == "__main__":
    main()
