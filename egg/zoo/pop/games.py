# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.interaction import LoggingStrategy
from egg.core.population import FullSweepAgentSampler, PopulationGame

from egg.zoo.pop.archs import (
    PopSymbolGame,
    EmSSLSender,
    Receiver,
)
from egg.zoo.emcom_as_ssl.losses import get_loss


def build_game(
    opts,
):  # Mat : remove opts.shared_vision (and opts.pretrain_vision?)
    loss = get_loss(
        temperature=opts.loss_temperature,
        similarity=opts.similarity,
        use_distributed_negatives=opts.use_distributed_negatives,
        loss_type=opts.loss_type,
    )

    train_logging_strategy = LoggingStrategy(False, False, True, True, True, False)
    test_logging_strategy = LoggingStrategy(False, False, True, True, True, False)

    if opts.simclr_sender:
        raise NotImplementedError("Not implemented in the pop Game")
    else:
        sender = EmSSLSender(
            vision_module=opts.sender_model_name,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.projection_output_dim,
            temperature=opts.gs_temperature,
            trainable_temperature=opts.train_gs_temperature,
            straight_through=opts.straight_through,
        )
        receiver = Receiver(
            vision_module=opts.rcv_model_name,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.projection_output_dim,
        )

    senders = [sender]
    receivers = [receiver]
    losses = [loss]
    train_logging_strategy = (train_logging_strategy,)
    test_logging_strategy = (test_logging_strategy,)
    game = PopSymbolGame(train_logging_strategy, test_logging_strategy)

    agents_loss_sampler = FullSweepAgentSampler(senders, receivers, losses)
    game = PopulationGame(game, agents_loss_sampler)

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
