# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.archs import (
    Receiver,
    Sender,
    VisionGameWrapper,
    VisionModule,
    get_vision_module,
)
from egg.zoo.simclr.losses import Loss


def build_game(
    batch_size: int = 32,
    loss_temperature: float = 0.1,
    vision_encoder_name: str = "resnet50",
    output_size: int = 128,
    is_distributed: bool = False,
):
    vision_module, visual_features_dim = get_vision_module(
        encoder_arch=vision_encoder_name
    )
    vision_encoder = VisionModule(vision_module=vision_module)

    train_logging_strategy = LoggingStrategy.minimal()
    assert (
        not batch_size % 2
    ), f"Batch size must be multiple of 2. Found {batch_size} instead"

    loss = Loss(batch_size, loss_temperature)

    sender = Sender(visual_features_dim=visual_features_dim, output_dim=output_size)
    receiver = Receiver(visual_features_dim=visual_features_dim, output_dim=output_size)

    game = SenderReceiverContinuousCommunication(
        sender, receiver, loss, train_logging_strategy
    )

    game = VisionGameWrapper(game, vision_encoder)
    if is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
