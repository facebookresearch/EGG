# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
from typing import Union

import numpy as np
import torch
from torchvision import datasets

from egg.core.batch import Batch
from egg.core.interaction import Interaction
from egg.zoo.population_game.data import ImageTransformation
from egg.zoo.population_game.games import build_game


def add_common_cli_args(parser):
    parser.add_argument(
        "--pretrain_vision",
        default=True,
        action="store_true",
        help="If set, pretrained vision modules will be used",
    )
    ###
    parser.add_argument(
        "--n_senders", type=int, default=3, help="Number of senders in the population"
    )
    parser.add_argument(
        "--n_recvs", type=int, default=3, help="Number of receivers in the population"
    )
    ###
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--dump_interaction_folder",
        type=str,
        default=None,
        help="Path where interaction will be saved. If None or empty string interaction won't be saved",
    )
    parser.add_argument(
        "--pdb", default=False, action="store_true", help="Run with pdb"
    )


def get_params(
    n_senders: bool,
    n_recvs: bool,
):
    params = dict(
        n_senders=n_senders,
        n_recvs=n_recvs,
    )

    distributed_context = argparse.Namespace(is_distributed=False)
    params_fixed = dict(
        pretrain_vision=True,
        shared_vision=True,
        use_augmentations=True,
        random_seed=111,
        model_name="resnet50",
        loss_temperature=1.0,
        similarity="cosine",
        projection_hidden_dim=2048,
        projection_output_dim=2048,
        gs_temperature=5.0,
        gs_temperature_decay=1.0,
        train_gs_temperature=False,
        straight_through=False,
        distributed_context=distributed_context,
    )
    params.update(params_fixed)

    params = argparse.Namespace(**params)

    random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.random_seed)
    return params


def get_game(params: argparse.Namespace, checkpoint_path: str):
    game = build_game(params, sampler="full")
    checkpoint = torch.load(checkpoint_path)
    game.load_state_dict(checkpoint.model_state_dict)
    return game


def save_interaction(interaction: Interaction, log_dir: Union[pathlib.Path, str]):
    dump_dir = pathlib.Path(log_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(interaction, dump_dir / "interactions_test_set.pt")


def get_test_data(
    dataset_dir: str = "/datasets01/imagenet_full_size/061417/train",
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentations: bool = False,
    return_original_image: bool = False,
    image_size: int = 32,
):

    transformations = ImageTransformation(
        image_size, use_augmentations, return_original_image, "imagenet"
    )
    dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transformations
    )
    # dataset = datasets.ImageFolder(dataset_dir, transform=transformations)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def evaluate(game, data, device, n_senders, n_recvs):
    mean_loss = 0.0
    interactions = []
    n_batches = 0
    if torch.cuda.is_available():
        game.cuda()
    game.eval()
    with torch.no_grad():
        for batch in data:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)
            for _ in range(n_senders * n_recvs):
                optimized_loss, interaction = game(*batch)

                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                interactions.append(interaction)
            n_batches += 1

    mean_loss /= n_batches * n_senders * n_recvs
    full_interaction = Interaction.from_iterable(interactions)
    return mean_loss.item(), full_interaction
