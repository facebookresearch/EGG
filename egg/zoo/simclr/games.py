# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from egg.core.continous_communication import (
    ContinuousLinearReceiver,
    ContinuousLinearSender,
    SenderReceiverContinuousCommunication
)
from egg.core.gs_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolGameGS,
    SymbolReceiverWrapper
)
from egg.core.interaction import LoggingStrategy
from egg.core.reinforce_wrappers import (
    RnnSenderReinforce,
    RnnReceiverDeterministic,
    SenderReceiverRnnReinforce,
)
from egg.zoo.simclr.archs import (
    get_vision_modules,
    Receiver,
    Sender,
    SenderGS,
    ReceiverGS,
    VisionGameWrapper,
    VisionModule
)
from egg.zoo.simclr.losses import Loss


def build_sender_receiver_rf(
    sender,
    receiver,
    vocab_size,
    max_len,
    sender_embed_dim,
    receiver_embed_dim,
    sender_hidden,
    receiver_hidden,
    sender_rnn_num_layers,
    receiver_rnn_num_layers,
    recurrent_cell,
):
    sender = RnnSenderReinforce(
        agent=sender,
        vocab_size=vocab_size,
        embed_dim=sender_embed_dim,
        hidden_size=sender_hidden,
        max_len=max_len,
        num_layers=sender_rnn_num_layers,
        cell=recurrent_cell
    )
    receiver = RnnReceiverDeterministic(
        receiver,
        vocab_size=vocab_size,
        embed_dim=receiver_embed_dim,
        hidden_size=receiver_hidden,
        cell=recurrent_cell,
        num_layers=receiver_rnn_num_layers
    )
    return sender, receiver


def build_sender_receiver_continuous(
    sender,
    receiver,
    sender_input_size,
    sender_output_size,
):
    sender = ContinuousLinearSender(
        agent=sender,
        encoder_input_size=sender_input_size,
        encoder_hidden_size=sender_output_size
    )
    receiver = ContinuousLinearReceiver(receiver)
    return sender, receiver


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

    sender = Sender(
        visual_features_dim=visual_features_dim,
        projection_dim=opts.projection_dim
    )
    # fix projection dim dimension etc, especially when it's continuous
    effective_projection_dim = opts.projection_dim if opts.projection_dim != -1 else visual_features_dim
    receiver = Receiver(
        visual_features_dim=effective_projection_dim,
        output_dim=opts.receiver_output_size
    )

    loss = Loss(opts.batch_size, opts.ntxent_tau)
    train_logging_strategy = LoggingStrategy.minimal()
    test_logging_strategy = LoggingStrategy.minimal()
    if opts.communication_channel == "continuous":
        game = build_sender_receiver_continuous(
            sender,
            receiver,
            sender_input_sisze=effective_projection_dim,
            sender_output_size=opts.sender_output_size,
        )
        game = SenderReceiverContinuousCommunication(
            sender,
            receiver,
            loss,
            train_logging_strategy,
            test_logging_strategy
        )
    elif opts.communication_channel == "rf":
        sender, receiver = build_sender_receiver_rf(
            sender,
            receiver,
            vocab_size=opts.vocab_size,
            max_len=opts.max_len,
            sender_embed_dim=opts.sender_embedding,
            receiver_embed_dim=opts.receiver_embedding,
            sender_rnn_hidden=effective_projection_dim,
            receiver_rnn_hidden=opts.receiver_rnn_hidden,
            sender_rnn_num_layers=opts.sender_rnn_num_layers,
            receiver_rnn_num_layers=opts.receiver_rnn_num_layers,
            cell=opts.recurrent_cell,
        )
        game = SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            train_logging_strategy=train_logging_strategy,
            test_logging_strategy=test_logging_strategy
        )
    elif opts.communication_channel == "gs":
        sender = SenderGS(
            visual_features_dim=visual_features_dim,
            vocab_size=opts.vocab_size
        )
        sender = GumbelSoftmaxWrapper(
            sender,
            temperature=1.0,  # TODO make it a param!!
            trainable_temperature=False,
            straight_through=False,
        )
        receiver = ReceiverGS(visual_features_dim, opts.receiver_output_size)
        receiver = SymbolReceiverWrapper(receiver, opts.vocab_size, opts.receiver_output_size)
        train_logging_strategy = LoggingStrategy(False, False, False, True, False, True)
        game = SymbolGameGS(
            sender,
            receiver,
            loss,
            train_logging_strategy=train_logging_strategy,
            test_logging_strategy=test_logging_strategy
        )
    else:
        raise NotImplementedError(f"Cannot recognize communication channel {opts.communication_channel}")

    game = VisionGameWrapper(game, vision_encoder)
    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
