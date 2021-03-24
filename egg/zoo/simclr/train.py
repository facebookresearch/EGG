# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb

import egg.core as core
from egg.zoo.simclr.data import get_dataloader
from egg.zoo.simclr.games import build_game
from egg.zoo.simclr.game_callbacks import get_callbacks
from egg.zoo.simclr.LARC import LARC
from egg.zoo.simclr.utils import add_weight_decay, get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(opts)
    if opts.wandb and opts.distributed_context.is_leader:
        import uuid
        id = str(uuid.uuid4())
        wandb.init(project="language-as-rl", id=id)
        wandb.config.update(opts)
    assert not opts.batch_size % 2, (
        f"Batch size must be multiple of 2. Found {opts.batch_size} instead"
    )
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)\n"
        f"Using dataset {opts.dataset_name} with image size: {opts.image_size}. "
        f"Applying augmentations: {opts.use_augmentations}\n"
    )
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader, validation_loader = get_dataloader(
        dataset_name=opts.dataset_name,
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        validation_dataset_dir=opts.validation_dataset_dir,
        num_workers=opts.num_workers,
        use_augmentations=opts.use_augmentations,
        imagenet_normalization=opts.dataset_name.lower() == "imagenet" or opts.pretrain_vision,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed
    )

    simclr_game = build_game(opts)
    if opts.wandb and opts.distributed_context.is_leader:
        wandb.watch(simclr_game, log="all")

    model_parameters = add_weight_decay(
        simclr_game,
        opts.weight_decay,
        skip_name='bn'
    )

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.n_epochs)

    if opts.distributed_context.is_distributed and opts.distributed_context.world_size > 2:
        optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

    callbacks = get_callbacks(
        opts=opts,
        agent=simclr_game.game.sender.gs_layer,
    )

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
        aggregate_interaction_logs=False
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys
    main(sys.argv[1:])
