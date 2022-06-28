# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from egg.zoo.pop.scripts.analysis_tools.test_game import initialize_classifiers
from egg.zoo.pop.utils import load_from_checkpoint
from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from egg.zoo.pop.archs import (
    AgentSampler,
    Game,
    PopulationGame,
    Receiver,
    Sender,
    ContinuousSender,
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

def build_senders_receivers(opts,vision_model_names_senders=None,vision_model_names_receiver=None):
    if vision_model_names_senders is None:
        vision_model_names_senders = opts.vision_model_names_senders
    if vision_model_names_receiver is None:
        vision_model_names_receiver = opts.vision_model_names_recvs

    vision_model_names_senders = eval(
        vision_model_names_senders.replace("#", '"')  # Mat : ...
    )
    vision_model_names_receiver = eval(
        vision_model_names_receiver.replace("#", '"')
    )

    # We don't do the single module thing here
    # if not (vision_model_names_senders and vision_model_names_receiver):
    #     vision_module_names_senders = eval(opts.vision_module_names.replace("#", '"'))
    #     vision_module_names_receivers = eval(opts.vision_module_names.replace("#", '"'))

    vision_modules_senders = [
        initialize_vision_module(name=vision_model_names_senders[i], pretrained=not opts.retrain_vision)
        if not opts.keep_classification_layer else initialize_classifiers(name=vision_model_names_senders[i], pretrained=not opts.retrain_vision)
        for i in range(len(vision_model_names_senders))
    ]
    vision_modules_receivers = [
        initialize_vision_module(name=vision_model_names_receiver[i], pretrained=not opts.retrain_vision)
        if not opts.keep_classification_layer else initialize_classifiers(name=vision_model_names_senders[i], pretrained=not opts.retrain_vision)
        for i in range(len(vision_model_names_receiver))
    ]
    if opts.continuous_com:
        senders = [
            ContinuousSender(
                vision_module=vision_modules_senders[i][0],
                input_dim=vision_modules_senders[i][1],
                vocab_size=opts.vocab_size,
                name=vision_model_names_senders[i],
                non_linearity=opts.non_linearity,
                force_gumbel=opts.force_gumbel,
                forced_gumbel_temperature=opts.gs_temperature,
            )
            for i in range(len(vision_model_names_senders))
        ]
        receivers = [
                Receiver(
                vision_module=vision_modules_receivers[i][0],
                input_dim=vision_modules_receivers[i][1],
                hidden_dim=opts.recv_hidden_dim,
                output_dim=opts.vocab_size,
                temperature=opts.recv_temperature,
                name=vision_model_names_receiver[i],
            )
            for i in range(len(vision_model_names_receiver))
        ]
    else:
        senders = [
            GumbelSoftmaxWrapper(
                Sender(
                    vision_module=vision_modules_senders[i][0],
                    input_dim=vision_modules_senders[i][1],
                    vocab_size=opts.vocab_size,
                    name=vision_model_names_senders[i],
                ),
                temperature=opts.gs_temperature,
                trainable_temperature=opts.train_gs_temperature,
                straight_through=opts.straight_through,
            )
            for i in range(len(vision_model_names_senders))
        ]
        receivers = [
            SymbolReceiverWrapper(
                Receiver(
                    vision_module=vision_modules_receivers[i][0],
                    input_dim=vision_modules_receivers[i][1],
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.vocab_size,
                    temperature=opts.recv_temperature,
                    name=vision_model_names_receiver[i],
                ),
                opts.vocab_size,
                opts.vocab_size, # probably useless ? ask Roberto
            )
            for i in range(len(vision_model_names_receiver))
        ]
    print(vision_model_names_senders)
    print(vision_model_names_receiver)
    return senders, receivers

def build_game(opts):
    train_logging_strategy = LoggingStrategy(
        False, False, False, False, False, False, False
    )
    test_logging_strategy = LoggingStrategy(False, False, True, True, True, True, False)

    senders, receivers = build_senders_receivers(opts)
    agents_loss_sampler = AgentSampler(
        senders,
        receivers,
        [loss],
    )

    game = Game(
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
        noisy=opts.noisy_channel,
    )

    game = PopulationGame(game, agents_loss_sampler)

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def build_second_game(opts):
    """temporary version of build_game, train a new receiver with a pretrained sender
    (might end up using checkpoints instead...)
    """

    pop_game = build_game(opts)
    load_from_checkpoint(pop_game, opts.base_checkpoint_path)

    for old_sender in  pop_game.agents_loss_sampler.senders:
        for param in old_sender.parameters():
            param.requires_grad = False
    for old_receiver in  pop_game.agents_loss_sampler.receivers:
        for param in old_receiver.parameters():
            param.requires_grad = False

    new_senders, new_receivers = build_senders_receivers(opts, opts.additional_senders, opts.additional_receivers)
    pop_game.agents_loss_sampler.avoid_training_old()
    pop_game.agents_loss_sampler.add_senders(new_senders)
    pop_game.agents_loss_sampler.add_receivers(new_receivers)

    # if opts.distributed_context.is_distributed:
    #     game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return pop_game



    