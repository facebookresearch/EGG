# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

from egg.core.interaction import Interaction
from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.data import ImageTransformation
from egg.zoo.emcom_as_ssl.games import build_game


def add_common_cli_args(parser):
    parser.add_argument(
        "--simclr_sender",
        default=False,
        action="store_true",
        help="Running gaussian evaluation loading a SimCLR model"
    )
    parser.add_argument(
        "--discrete_evaluation_simclr",
        default=False,
        action="store_true",
        help="Evaluate SimCLR playing the game discretizing the message_like layer"
    )
    parser.add_argument(
        "--shared_vision",
        default=False,
        action="store_true",
        help="Load a model with shared vision module"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="xent",
        choices=["xent", "ntxent"],
        help="Specify loss used to train the model"
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        help="Path to tes set to use for evaluation"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--evaluate_with_augmentations",
        default=False,
        action="store_true",
        help="Running gaussian evaluation with data augmentation"
    )
    parser.add_argument(
        "--dump_interaction_folder",
        type=str,
        default=None,
        help="Path where interaction will be saved. If None or empty string interaction won't be saved"
    )
    parser.add_argument(
        "--pdb",
        default=False,
        action="store_true",
        help="Run with pdb"
    )


def get_params(
    simclr_sender: bool,
    shared_vision: bool,
    discrete_evaluation_simclr: bool,
    loss_type: str
):
    params = dict(
        simclr_sender=simclr_sender,
        loss_type=loss_type,
        shared_vision=shared_vision,
        discrete_evaluation_simclr=discrete_evaluation_simclr
    )

    distributed_context = argparse.Namespace(is_distributed=False)
    params_fixed = dict(
        use_augmentations=True,
        random_seed=111,
        model_name="resnet50",
        loss_temperature=1.0,
        pretrain_vision=False,
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
    game = build_game(params)
    checkpoint = torch.load(checkpoint_path)
    game.load_state_dict(checkpoint.model_state_dict)
    return game


def save_interaction(
    interaction: Interaction,
    log_dir: Union[pathlib.Path, str]
):
    dump_dir = pathlib.Path(log_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(interaction, dump_dir / f"interactions_test_set.pt")


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmentations: bool = False,
):
    transformations = ImageTransformation(image_size, use_augmentations, False)

    dataset = datasets.ImageFolder(
        dataset_dir,
        transform=transformations
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def evaluate(
    game: nn.Module,
    data: torch.utils.data.DataLoader,
):
    if torch.cuda.is_available():
        game.cuda()
    game.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_loss = 0.0
    interactions = []
    n_batches = 0
    soft_accuracy, game_accuracy = 0.0, 0.0
    with torch.no_grad():
        for batch in data:
            batch = move_to(batch, device)
            optimized_loss, interaction = game(*batch)

            interaction = interaction.to("cpu")
            interactions.append(interaction)

            mean_loss += optimized_loss
            soft_accuracy += interaction.aux['acc'].mean().item()
            game_accuracy += interaction.aux['game_acc'].mean().item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"finished batch {n_batches}")
            # when running kmeans, we first feed the train data.
            # given we're clustering only a subset of 100_000 elements from the
            # training data we can stop after 128 (bsz) X 800 (batches) = 102_400 samples
            if n_batches == 800:
                break

    print(f"processed {n_batches} batches in total")
    mean_loss /= n_batches
    soft_accuracy /= n_batches
    game_accuracy /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    return mean_loss, soft_accuracy, game_accuracy, full_interaction
