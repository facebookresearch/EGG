# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import pathlib

import torch

import egg.core as core
from egg.core.interaction import Interaction
from egg.core.util import move_to
from egg.zoo.simclr_replica.data import get_random_noise_dataloader


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_opts(params):
    parser = argparse.ArgumentParser()

    # Data opts
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/datasets01/imagenet_full_size/061417/train",
        help="Dataset location",
    )
    parser.add_argument(
        "--validation_dataset_dir",
        type=str,
        default="",
        help="Dataset location",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imagenet",
        choices=["cifar10", "imagenet"],
        help="Dataset used for training",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Workers used in the dataloader"
    )

    # Vision module opts
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )

    # Loss opts
    parser.add_argument(
        "--ntxent_tau",
        type=float,
        default=0.1,
        help="Temperature for NT XEnt loss",
    )

    # Arch opts
    parser.add_argument(
        "--output_size",
        type=int,
        default=128,
        help="Sender/Receiver output size"
    )

    # Misc opts
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    parser.add_argument(
        "--gaussian_noise_evaluation",
        action="store_true",
        default=False,
        help="Perform and evaluation on gaussian noise at the end of training and store the interaction output"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Run the game with wandb logging"
    )

    opts = core.init(arg_parser=parser, params=params)
    return opts


def noise_eval(
    game,
    validation_data,
    is_distributed,
    device,
):
    mean_loss = 0.0
    interactions = []
    n_batches = 0
    game.eval()
    with torch.no_grad():
        for batch in validation_data:
            assert len(batch[0]) == 3
            original_image = batch[0][-1]
            batch[0] = batch[0][:-1]
            batch = move_to(batch, device)
            batch[0].append(original_image)

            optimized_loss, interaction = game(*batch)

            if is_distributed:
                interaction = Interaction.gather_distributed_interactions(interaction)

            interaction = interaction.to("cpu")
            mean_loss += optimized_loss

            interactions.append(interaction)
            n_batches += 1

    mean_loss /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    return mean_loss.item(), full_interaction


def dump_noise_interaction(logs: Interaction, dump_dir: str):
    dump_dir = pathlib.Path(dump_dir) / "gaussian_noise"
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(logs, dump_dir / "interaction_noise")


def perform_gaussian_noise_evaluation(
    game,
    batch_size,
    distributed_context,
    seed,
    device,
    checkpoint_dir
):

    noise_loader = get_random_noise_dataloader(
        batch_size=batch_size,
        is_distributed=distributed_context.is_distributed,
        seed=seed
    )

    loss, noise_interaction = noise_eval(
        game=game,
        validation_data=noise_loader,
        is_distributed=distributed_context.is_distributed,
        device=device
    )

    if distributed_context.is_leader:
        dump = dict(val_gaussian_loss=loss, val_gaussian_acc=noise_interaction.aux["acc"].mean().item())
        output_message = json.dumps(dump)
        print(output_message, flush=True)

        dump_noise_interaction(noise_interaction, checkpoint_dir)
