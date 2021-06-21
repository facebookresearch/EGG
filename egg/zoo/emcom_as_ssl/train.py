# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import egg.core as core
from egg.zoo.emcom_as_ssl.data import get_dataloader
from egg.zoo.emcom_as_ssl.game_callbacks import get_callbacks
from egg.zoo.emcom_as_ssl.games import build_game
from egg.zoo.emcom_as_ssl.LARC import LARC
from egg.zoo.emcom_as_ssl.utils import add_weight_decay, get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    print(f"{opts}\n")
    assert (
        not opts.batch_size % 2
    ), f"Batch size must be multiple of 2. Found {opts.batch_size} instead"
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)\n"
        f"Applying augmentations: {opts.use_augmentations} with image size: {opts.image_size}.\n"
    )
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        dataset_name=opts.dataset_name,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
        use_augmentations=opts.use_augmentations,
        return_original_image=opts.return_original_image,
    )

    game = build_game(opts)

    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    if (
        opts.distributed_context.is_distributed
        and opts.distributed_context.world_size > 2
        and opts.use_larc
    ):
        optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

    callbacks = get_callbacks(
        shared_vision=opts.shared_vision,
        n_epochs=opts.n_epochs,
        checkpoint_dir=opts.checkpoint_dir,
        sender=game.game.sender,
        train_gs_temperature=opts.train_gs_temperature,
        minimum_gs_temperature=opts.minimum_gs_temperature,
        update_gs_temp_frequency=opts.update_gs_temp_frequency,
        gs_temperature_decay=opts.gs_temperature_decay,
        is_distributed=opts.distributed_context.is_distributed,
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
