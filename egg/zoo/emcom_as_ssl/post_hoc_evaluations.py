# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from egg.core.interaction import Interaction
from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.data import get_dataloader
from egg.zoo.emcom_as_ssl.gaussian_noise_data import get_random_noise_dataloader


def aggregate_print(loss: float, logs: Interaction, mode: str):
    dump = dict(loss=loss)
    aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())

    dump.update(aggregated_metrics)
    dump.update(dict(mode=mode))

    output_message = json.dumps(dump)
    print(output_message, flush=True)


def eval_print_result_and_store_interactions(
    game: nn.Module,
    data: DataLoader,
    device: torch.device,
    is_distributed: bool,
    mode: str,
    log_dir: str,
    rank: int,
    use_wandb: bool = False,
):
    mean_loss = 0.0
    interactions = []
    n_batches = 0
    game.eval()
    with torch.no_grad():
        for batch in data:
            batch = move_to(batch, device)
            optimized_loss, interaction = game(*batch)

            if is_distributed:
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")
            interactions.append(interaction)

            mean_loss += optimized_loss
            n_batches += 1

    mean_loss /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    if rank == 0:
        dump_dir = pathlib.Path(log_dir) / mode
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(full_interaction, dump_dir / "interactions.pt")

    aggregate_print(
        loss=mean_loss.item(),
        logs=full_interaction,
        mode=mode
    )
    if use_wandb:
        wandb.log({
            f"post_hoc_{mode}_loss": mean_loss.item(),
            f"post_hoc_{mode}_soft_accuracy": full_interaction.aux['acc'].mean().item(),
            f"post_hoc_{mode}_game_accuracy": full_interaction.aux['game_acc'].mean().item(),
        })


def post_hoc_evaluations(
    game: nn.Module,
    device: torch.device,
    log_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentations: bool = False,
    gaussian_noise_dataset_size: int = 3072,
    is_distributed: bool = False,
    rank: int = -1,
    use_wandb: bool = False,
    seed: int = 111,
):
    assert rank >= -1
    o_test_path = (
        "/private/home/mbaroni/agentini/representation_learning/"
        "generalizaton_set_construction/80_generalization_data_set/"
    )

    o_test_loader, _ = get_dataloader(
        dataset_dir=o_test_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmentations=use_augmentations,
        is_distributed=is_distributed,
        seed=seed
    )

    eval_print_result_and_store_interactions(
        game=game,
        data=o_test_loader,
        device=device,
        is_distributed=is_distributed,
        mode="o_test",
        log_dir=log_dir,
        rank=rank,
        use_wandb=use_wandb
    )

    gaussian_noise_data = get_random_noise_dataloader(
        dataset_size=gaussian_noise_dataset_size,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_augmentations=use_augmentations,
        is_distributed=is_distributed,
        seed=seed
    )
    eval_print_result_and_store_interactions(
        game=game,
        data=gaussian_noise_data,
        device=device,
        is_distributed=is_distributed,
        mode="gaussian_noise",
        log_dir=log_dir,
        rank=rank,
        use_wandb=use_wandb
    )
