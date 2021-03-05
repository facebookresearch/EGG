# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.simclr.data import get_dataloader
from egg.zoo.simclr.games import build_game
from egg.zoo.simclr.game_callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter,
    VisionModelSaver
)
from egg.zoo.simclr.utils import get_opts


def main(params):
    opts = get_opts(params=params)
    batch_size = opts.batch_size // opts.distributed_context.world_size
    val_batch_size = opts.val_batch_size // opts.distributed_context.world_size
    assert (not batch_size % 2) and (not val_batch_size % 2), (
        f"Batch size must be multiple of 2. Effective train_bsz is {opts.batch_size}, val bsz is {opts.val_batch_size} "
        f"split in opts.distributed_{opts.distributed_context.world_size} yielding {batch_size} samples per process"
    )
    if (not opts.distributed_context.is_distributed) or opts.distributed_context.local_rank == 0:
        print(opts)
        print(
            f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
            f"World size is {opts.distributed_context.world_size}\n"
            f"Using dataset {opts.dataset_name} with image size: {opts.image_size}. "
            f"Applying augmentations: {opts.use_augmentations}\n"
            f"Using batch of size {opts.batch_size} partioned on {opts.distributed_context.world_size} device(s)"
        )
        if opts.pdb:
            breakpoint()

    train_loader, validation_loader = get_dataloader(
        dataset_name=opts.dataset_name,
        train_dataset_dir=opts.train_dataset_dir,
        image_size=opts.image_size,
        batch_size=batch_size,  # effective batch size is batch_size * world_size
        val_batch_size=val_batch_size,
        do_validation=opts.do_validation,
        val_dataset_dir=opts.val_dataset_dir,
        num_workers=opts.num_workers,
        use_augmentations=opts.use_augmentations,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed
    )

    simclr_game = build_game(opts)

    optimizer = core.build_optimizer(simclr_game.parameters())

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(opts.early_stopping_thr, validation=False),
        BestStatsTracker(),
        VisionModelSaver(opts.shared_vision)
    ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
