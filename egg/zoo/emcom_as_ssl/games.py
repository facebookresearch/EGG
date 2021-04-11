# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.interaction import LoggingStrategy
from egg.zoo.emcom_as_ssl.archs import (
    EmComSSLSymbolGame,
    EmSSLSender,
    get_vision_modules,
    Receiver,
    SimCLRSender,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.emcom_as_ssl.losses import get_loss


def build_vision_encoder(
    model_name: str = "resnet50",
    shared_vision: bool = False,
    pretrain_vision: bool = False
):
    sender_vision_module, receiver_vision_module, visual_features_dim = get_vision_modules(
        encoder_arch=model_name,
        shared=shared_vision,
        pretrain_vision=pretrain_vision
    )
    vision_encoder = VisionModule(
        sender_vision_module=sender_vision_module,
        receiver_vision_module=receiver_vision_module
    )
    return vision_encoder, visual_features_dim


def build_game(opts):
    vision_encoder, visual_features_dim = build_vision_encoder(
        model_name=opts.model_name,
        shared_vision=opts.shared_vision,
        pretrain_vision=opts.pretrain_vision
    )

    loss = get_loss(
        temperature=opts.loss_temperature,
        similarity=opts.similarity,
        loss_type=opts.loss_type
    )

    train_logging_strategy = LoggingStrategy(False, False, True, True, True, False)
    test_logging_strategy = LoggingStrategy(False, False, True, True, True, False)

    if opts.simclr_sender:
        sender = SimCLRSender(
            input_dim=visual_features_dim,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.projection_output_dim,
        )
    else:
        sender = EmSSLSender(
            input_dim=visual_features_dim,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.projection_output_dim,
            temperature=opts.gs_temperature,
            trainable_temperature=opts.train_gs_temperature,
            straight_through=opts.straight_through,
        )
    receiver = Receiver(
        input_dim=visual_features_dim,
        hidden_dim=opts.projection_hidden_dim,
        output_dim=opts.projection_output_dim
    )

    game = EmComSSLSymbolGame(
        sender,
        receiver,
        loss,
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy
    )

    game = VisionGameWrapper(game, vision_encoder)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
