# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb

import egg.core as core
from egg.zoo.simclr_replica.data import get_dataloader
from egg.zoo.simclr_replica.games import build_game
from egg.zoo.simclr_replica.game_callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter,
    VisionModelSaver,
    WandbLogger
)
from egg.zoo.simclr_replica.LARC import LARC
from egg.zoo.simclr_replica.utils import add_weight_decay, get_opts, perform_gaussian_noise_evaluation


def main(params):
    opts = get_opts(params=params)
    print(opts)
    if opts.wandb and opts.distributed_context.is_leader:
        if opts.checkpoint_dir:
            id = opts.checkpoint_dir.split("/")[-1]
        else:
            import uuid
            id = str(uuid.uuid4())
        wandb.init(project="vis-emcomm", id=id)
        wandb.config.update(opts)
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}\n"
        f"Using imagenet with image size: {opts.image_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)"
    )
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader, validation_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        dataset_name=opts.dataset_name,
        validation_dataset_dir=opts.validation_dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed
    )

    simclr_game = build_game(
        batch_size=opts.batch_size,
        loss_temperature=opts.ntxent_tau,
        vision_encoder_name=opts.model_name,
        output_size=opts.output_size,
        is_distributed=opts.distributed_context.is_distributed
    )
    if opts.wandb and opts.distributed_context.is_leader:
        wandb.watch(simclr_game, log="all")

    model_parameters = add_weight_decay(
        simclr_game,
        opts.weight_decay,
        skip_name='bn'
    )

    optimizer_original = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer = LARC(optimizer_original, trust_coefficient=0.001, clip=False, eps=1e-8)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_original, T_max=opts.n_epochs)

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
        VisionModelSaver(),
        core.InteractionSaver(
            train_epochs=[1, opts.n_epochs],
            test_epochs=[1, opts.n_epochs],
            checkpoint_dir=opts.checkpoint_dir
        ),
        WandbLogger()
    ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)

    if opts.gaussian_noise_evaluation:
        perform_gaussian_noise_evaluation(
            game=simclr_game,
            batch_size=opts.batch_size,
            distributed_context=opts.distributed_context,
            seed=opts.random_seed,
            device=opts.device,
            checkpoint_dir=opts.checkpoint_dir
        )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
