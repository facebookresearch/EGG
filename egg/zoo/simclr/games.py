# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.continous_communication import (
    ContinuousLinearSender,
    SenderReceiverContinuousCommunication
)
from egg.core.interaction import LoggingStrategy
from egg.core.reinforce_wrappers import (
    RnnSenderReinforce,
    SenderReceiverRnnReinforce,
)
from egg.zoo.simclr.archs import (
    get_vision_modules,
    Receiver,
    RnnReceiverDeterministicContrastive,
    Sender,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.simclr.losses import Loss


def build_game(opts):
    device = torch.device("cuda" if opts.cuda else "cpu")
    sender_vision_module, receiver_vision_module, visual_features_dim = get_vision_modules(
        encoder_arch=opts.model_name,
        shared=opts.shared_vision,
        pretrain_vision=opts.pretrain_vision
    )
    vision_encoder = VisionModule(
        seneder_vision_module=sender_vision_module,
        receiver_vision_module=receiver_vision_module
    )

    train_logging_strategy = LoggingStrategy.minimal()
    batch_size = opts.batch_size // opts.distributed_context.world_size
    assert not batch_size % 2, (
        f"Batch size must be multiple of 2. Effective bsz is {opts.batch_size} split "
        f"in opts.distributed_{opts.distributed_context.world_size} yielding {batch_size} samples per process"
    )

    loss = Loss(batch_size, opts.ntxent_tau, device)

    sender = Sender(
        visual_features_dim=visual_features_dim,
        projection_dim=opts.projection_dim
    )
    effective_projection_dim = opts.projection_dim if opts.projection_dim != -1 else visual_features_dim
    if opts.communication_channel == "rf":
        sender = RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=effective_projection_dim,
            max_len=opts.max_len,
            num_layers=opts.sender_rnn_num_layers,
            cell=opts.sender_cell
        )
        receiver = Receiver(
            msg_input_dim=opts.receiver_rnn_hidden,
            img_feats_input_dim=effective_projection_dim,
            output_dim=opts.receiver_output_size
        )
        receiver = RnnReceiverDeterministicContrastive(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_rnn_hidden,
            cell=opts.receiver_cell,
            num_layers=opts.receiver_num_layers
        )
        game = SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            train_logging_strategy=train_logging_strategy
        )
    elif opts.communication_channel == "continuous":
        sender = ContinuousLinearSender(
            agent=sender,
            encoder_input_size=effective_projection_dim,
            encoder_hidden_size=opts.sender_output_size
        )
        receiver = Receiver(
            msg_input_dim=opts.sender_output_size,
            img_feats_input_dim=effective_projection_dim,
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
