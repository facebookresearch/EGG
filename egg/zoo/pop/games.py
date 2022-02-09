# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from egg.zoo.pop.archs import (
    AgentSampler,
    Game,
    PopulationGame,
    Receiver,
    Sender,
    initialize_vision_module,
)


def loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):
    labels = torch.arange(receiver_output.shape[0], device=receiver_output.device)
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def build_game(opts):

    train_logging_strategy = LoggingStrategy(
        False, False, False, False, False, False, False
    )
    test_logging_strategy = LoggingStrategy(False, False, True, True, True, True, False)

    if opts.use_different_architectures:
        vision_module_names_senders = eval(
            opts.vision_model_names_senders.replace("#", '"')  # Mat : ...
        )
        vision_module_names_receivers = eval(
            opts.vision_model_names_recvs.replace("#", '"')
        )

        if not (vision_module_names_senders and vision_module_names_receivers):
            vision_module_names_senders = eval(
                opts.vision_module_names.replace("#", '"')
            )
            vision_module_names_receivers = eval(
                opts.vision_module_names.replace("#", '"')
            )

        print(vision_module_names_senders)
        print(vision_module_names_receivers)

        vision_modules_senders = [
            initialize_vision_module(
                name=vision_module_names_senders[i], pretrained=True
            )
            for i in range(opts.n_senders)
        ]
        vision_modules_receivers = [
            initialize_vision_module(
                name=vision_module_names_receivers[i], pretrained=True
            )
            for i in range(opts.n_recvs)
        ]

        senders = [
            GumbelSoftmaxWrapper(
                Sender(
                    vision_module=vision_modules_senders[i][0],
                    input_dim=vision_modules_senders[i][1],
                    vocab_size=opts.vocab_size,
                    name=vision_module_names_senders[i],
                ),
                temperature=opts.gs_temperature,
                trainable_temperature=opts.train_gs_temperature,
                straight_through=opts.straight_through,
            )
            for i in range(opts.n_senders)
        ]
        receivers = [
            SymbolReceiverWrapper(
                Receiver(
                    vision_module=vision_modules_receivers[i][0],
                    input_dim=vision_modules_receivers[i][1],
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.recv_output_dim,
                    temperature=opts.recv_temperature,
                    name=vision_module_names_receivers[i],
                ),
                opts.vocab_size,
                opts.recv_output_dim,
            )
            for i in range(opts.n_recvs)
        ]

    else:
        vision_module, input_dim, name = initialize_vision_module(
            name=opts.vision_model_name, pretrained=True
        )
        senders = [
            GumbelSoftmaxWrapper(
                Sender(
                    vision_module=vision_module,
                    input_dim=input_dim,
                    vocab_size=opts.vocab_size,
                    name=name,
                ),
                temperature=opts.gs_temperature,
                trainable_temperature=opts.train_gs_temperature,
                straight_through=opts.straight_through,
            )
            for _ in range(opts.n_senders)
        ]
        receivers = [
            SymbolReceiverWrapper(
                Receiver(
                    vision_module=vision_module,
                    input_dim=input_dim,
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.recv_output_dim,
                    temperature=opts.recv_temperature,
                    name=name,
                ),
                opts.vocab_size,
                opts.recv_output_dim,
            )
            for _ in range(opts.n_recvs)
        ]

    agents_loss_sampler = AgentSampler(
        senders,
        receivers,
        [loss],
    )

    game = Game(
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    game = PopulationGame(game, agents_loss_sampler)

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def _build_second_game(opts):
    """temporary version of build_game, train a new receiver with a pretrained sender"""
    train_logging_strategy = LoggingStrategy(
        False, False, False, False, False, False, False
    )
    test_logging_strategy = LoggingStrategy(False, False, True, True, True, True, False)

    if opts.use_different_architectures: # Mat : Nearly forgot that one twice. Maybe refactor the opts
        vision_module_names_senders = eval(
            opts.vision_model_names_senders.replace("#", '"')
        )
        vision_module_names_receivers = eval(
            opts.vision_model_names_recvs.replace("#", '"')
        )

        print(vision_module_names_senders)
        print(vision_module_names_receivers)

        vision_modules_senders = [ # Mat : input are paths to pretrained models
            torch.load(vision_module_names_senders[i])
            for i in range(opts.n_senders)
        ]
        vision_modules_receivers = [
            initialize_vision_module(
                name=vision_module_names_receivers[i], pretrained=True
            )
            for i in range(opts.n_recvs)
        ]

        senders = [
            GumbelSoftmaxWrapper(
                # Mat : do smthing here
                Sender(
                    vision_module=vision_modules_senders[i][0],
                    input_dim=vision_modules_senders[i][1],
                    vocab_size=opts.vocab_size,
                    name=vision_module_names_senders[i],
                ),
                temperature=opts.gs_temperature,
                trainable_temperature=opts.train_gs_temperature,
                straight_through=opts.straight_through,
            )
            for i in range(opts.n_senders)
        ]
        receivers = [
            SymbolReceiverWrapper(
                Receiver(
                    vision_module=vision_modules_receivers[i][0],
                    input_dim=vision_modules_receivers[i][1],
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.recv_output_dim,
                    temperature=opts.recv_temperature,
                    name=vision_module_names_receivers[i],
                ),
                opts.vocab_size,
                opts.recv_output_dim,
            )
            for i in range(opts.n_recvs)
        ]

    else:
        vision_module, input_dim, name = initialize_vision_module(
            name=opts.vision_model_name, pretrained=True
        )
        senders = [
            GumbelSoftmaxWrapper(
                Sender(
                    vision_module=vision_module,
                    input_dim=input_dim,
                    vocab_size=opts.vocab_size,
                    name=name,
                ),
                temperature=opts.gs_temperature,
                trainable_temperature=opts.train_gs_temperature,
                straight_through=opts.straight_through,
            )
            for _ in range(opts.n_senders)
        ]
        receivers = [
            SymbolReceiverWrapper(
                Receiver(
                    vision_module=vision_module,
                    input_dim=input_dim,
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.recv_output_dim,
                    temperature=opts.recv_temperature,
                    name=name,
                ),
                opts.vocab_size,
                opts.recv_output_dim,
            )
            for _ in range(opts.n_recvs)
        ]

    agents_loss_sampler = AgentSampler(
        senders,
        receivers,
        [loss],
    )

    game = Game(
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
    )

    game = PopulationGame(game, agents_loss_sampler)

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
