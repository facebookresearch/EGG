# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.continous_communication import (
    SenderReceiverContinuousCommunication
)
from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.contrastive.archs import (
    get_vision_module,
    Sender,
    Receiver,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.simclr.contrastive.losses import Loss


def build_game(opts):
    vision_module, visual_features_dim = get_vision_module(encoder_arch=opts.model_name)
    vision_encoder = VisionModule(vision_module=vision_module)

    train_logging_strategy = LoggingStrategy.minimal()
    assert not opts.batch_size % 2, (
        f"Batch size must be multiple of 2. Found {opts.batch_size} instead"
    )

    loss = Loss(opts.batch_size, opts.ntxent_tau)

    sender = Sender(
        visual_features_dim=visual_features_dim,
        output_dim=opts.output_size
    )
    receiver = Receiver(
        visual_features_dim=visual_features_dim,
        output_dim=opts.output_size
    )

    game = SenderReceiverContinuousCommunication(
        sender,
        receiver,
        loss,
        train_logging_strategy
    )

    game = VisionGameWrapper(game, vision_encoder)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
