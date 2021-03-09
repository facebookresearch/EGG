# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from egg.core.continous_communication import (
    SenderReceiverContinuousCommunication
)
from egg.core.interaction import LoggingStrategy
from egg.zoo.simclr.contrastive.archs import (
    get_vision_module,
    Receiver,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.simclr.contrastive.losses import Loss


def build_game(opts):
    vision_module, visual_features_dim = get_vision_module(encoder_arch=opts.model_name)
    vision_encoder = VisionModule(vision_module=vision_module)

    train_logging_strategy = LoggingStrategy.minimal()
    batch_size = opts.batch_size // opts.distributed_context.world_size
    assert not batch_size % 2, (
        f"Batch size must be multiple of 2. Effective bsz is {opts.batch_size} split "
        f"in opts.distributed_{opts.distributed_context.world_size} yielding {batch_size} samples per process"
    )

    loss = Loss(batch_size, opts.ntxent_tau, torch.device("cuda" if opts.cuda else "cpu"))

    sender = nn.Identity()
    receiver = Receiver(
        visual_features_dim=visual_features_dim,
        output_dim=opts.receiver_output_size
    )

    game = SenderReceiverContinuousCommunication(
        sender,
        receiver,
        loss,
        train_logging_strategy
    )

    game = VisionGameWrapper(game, vision_encoder)
    return game
