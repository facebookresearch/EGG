# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Here we rewrite the game to add senders and receivers one by one into a population instead of all at once.

import torch
import egg.core as core
from egg.core import ConsoleLogger
import argparse

# from egg.core.callbacks import WandbLogger
from egg.zoo.pop.data import get_dataloader
from egg.zoo.pop.game_callbacks import (
    BestStatsTracker,
    WandbLogger,
)
from egg.zoo.pop.games import build_game, build_senders_receivers
from egg.zoo.pop.utils import add_weight_decay, get_common_opts

from pathlib import Path
import os

def launch_partial_training(
    game: torch.nn.Module,
    opts: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Launch partial training for a game with the given options and data loaders.

    :param game: The game to train.
    :param opts: Training options.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    """
    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.Adam(model_parameters, lr=opts.lr)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
        WandbLogger(opts),
    ]

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)

def main(params):
    # normal first training for all agents
    opts = get_common_opts(params=params)
    print(opts)

    # only keep the first sender and receiver for the first round of training
    senders=eval(opts.vision_model_names_senders.replace("#", '"'))
    receivers=eval(opts.vision_model_names_recvs.replace("#", '"'))
    opts.vision_model_names_senders = str(senders[:1])
    opts.vision_model_names_recvs = str(receivers[:1])
    game = build_game(opts)

    # deal with opts issues due to submitit module being replaced by sweep.py
    # all checkpoint_dirs from a sweep were stored in the same folder, causing overwriting
    # TODO : check for necessity before applying
    opts.checkpoint_dir = Path(opts.checkpoint_dir) / os.environ["SLURM_JOB_ID"]
    opts.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    assert (
        not opts.batch_size % 2
    ), f"Batch size must be multiple of 2. Found {opts.batch_size} instead"
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)\n"
        f"Applying augmentations: {opts.use_augmentations} with image size: {opts.image_size}.\n"
    )

    val_loader, train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        dataset_name=opts.dataset_name,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
        use_augmentations=opts.use_augmentations,
        return_original_image=opts.return_original_image,
        split_set=True,
        similbatch_training=opts.similbatch_training,
        shuffle=opts.shuffle,
    )

    launch_partial_training(game, opts, train_loader, val_loader)
    for sender, receiver in zip(senders[1:], receivers[1:]):
        print(f"Adding sender {sender}")
        _sender, _ = build_senders_receivers(opts, str([sender]), None)
        game.agents_loss_sampler.add_senders(_sender)
        launch_partial_training(game, opts, train_loader, val_loader)
    
        print(f"Adding receiver {receiver}")
        _, _receiver = build_senders_receivers(opts, None, str([receiver]))
        game.agents_loss_sampler.add_receivers(_receiver)
        
        launch_partial_training(game, opts, train_loader, val_loader)
    
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
