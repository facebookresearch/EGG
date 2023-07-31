# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from egg.zoo.pop.scripts.analysis_tools.test_game import initialize_classifiers
from egg.zoo.pop.utils import load_from_checkpoint
from egg.core.gs_wrappers import (
    GumbelSoftmaxWrapper,
    SymbolReceiverWrapper,
    RnnSenderGS,
    RnnReceiverGS,
    SenderReceiverRnnGS,
)
from egg.core.reinforce_wrappers import (
    ReinforceWrapper,
    RnnSenderReinforce,
)
from egg.core.interaction import LoggingStrategy
from egg.zoo.pop.archs import (
    AgentSampler,
    Game,
    PopulationGame,
    Receiver,
    Sender,
    ContinuousSender,
    initialize_vision_module,
    RnnReceiverReinforce,
)
from egg.zoo.pop.scripts.simplicial import SimplicialWrapper, Empty_wrapper


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


def find_module_from_name(modules, name):
    for module in modules:
        if module[2] == name:
            return module


def build_senders_receivers(
    opts, vision_model_names_senders=None, vision_model_names_receiver=None
):
    if vision_model_names_senders is None:
        vision_model_names_senders = opts.vision_model_names_senders
    if vision_model_names_receiver is None:
        vision_model_names_receiver = opts.vision_model_names_recvs

    vision_model_names_senders = eval(
        vision_model_names_senders.replace("#", '"')  # Mat : ...
    )
    vision_model_names_receiver = eval(vision_model_names_receiver.replace("#", '"'))

    vision_modules = [
        initialize_vision_module(
            name=module_name,
            pretrained=not opts.retrain_vision,
            aux_logits=not opts.remove_auxlogits,
        )
        if not opts.keep_classification_layer
        else initialize_classifiers(
            name=module_name,
            pretrained=not opts.retrain_vision,
            aux_logits=not opts.remove_auxlogits,
        )
        for module_name in set(vision_model_names_senders + vision_model_names_receiver)
    ]

    if opts.com_channel == "continuous":  # legacy parameter :/ TODO Matéo: remove
        if opts.max_len != 1:
            raise NotImplementedError(
                f"Continuous communication channel only supports max_len=1, got {opts.max_len}"
            )
        senders = [
            ContinuousSender(
                vision_module=find_module_from_name(vision_modules, module_name)[0],
                input_dim=find_module_from_name(vision_modules, module_name)[
                    1
                ],  # TODO Matéo: repeated twice :/ think about optimising
                vocab_size=opts.vocab_size,
                name=module_name,
                non_linearity=opts.non_linearity,
                force_gumbel=opts.force_gumbel,
                forced_gumbel_temperature=opts.gs_temperature,
                block_com_layer=opts.block_com_layer,
            )
            for module_name in vision_model_names_senders
        ]
        receivers = [
            Receiver(
                vision_module=find_module_from_name(vision_modules, module_name)[0],
                input_dim=find_module_from_name(vision_modules, module_name)[1],
                hidden_dim=opts.recv_hidden_dim,
                output_dim=opts.vocab_size,
                temperature=opts.recv_temperature,
                name=module_name,
                block_com_layer=opts.block_com_layer,
            )
            for module_name in vision_model_names_receiver
        ]
    elif opts.com_channel == "simplicial":
        # assert vision_model_names_senders == vision_model_names_receiver == 1, "For now simplicial communication channel only supports one sender and one receiver"
        senders = [
            SimplicialWrapper(
                vision_module=find_module_from_name(vision_modules, module_name)[0],
                v_output_dim=find_module_from_name(vision_modules, module_name)[1],
                hidden_size=opts.vocab_size,
                L=opts.simplicial_L,
                temperature=opts.simplicial_temperature,
            )
            for module_name in vision_model_names_senders
        ]
        receivers = [
            Receiver(
                vision_module=find_module_from_name(vision_modules, module_name)[0],
                input_dim=find_module_from_name(vision_modules, module_name)[1],
                hidden_dim=opts.recv_hidden_dim,
                output_dim=opts.vocab_size,
                temperature=opts.recv_temperature,
                name=module_name,
                block_com_layer=opts.block_com_layer,
            )
            for module_name in vision_model_names_receiver
        ]
    # select communication channel wrapper
    else:
        if opts.com_channel == "gs":
            wrapper = GumbelSoftmaxWrapper
            rwrapper = SymbolReceiverWrapper
            kwargs = {
                "temperature": opts.gs_temperature,
                "trainable_temperature": opts.train_gs_temperature,
                "straight_through": opts.straight_through,
            }
            rkwargs = {
                "vocab_size": opts.vocab_size,
                "agent_input_size": opts.vocab_size,
            }
        elif opts.com_channel == "reinforce":
            wrapper = ReinforceWrapper
            rwrapper = SymbolReceiverWrapper
            kwargs = {}
            rkwargs = {
                "vocab_size": opts.vocab_size,
                "agent_input_size": opts.vocab_size,
            }
        elif opts.com_channel == "lstm":
            # multisymbol wrapper
            # TODO: we should be able to chose between gs and reinforce
            wrapper = RnnSenderGS
            rwrapper = RnnReceiverGS
            kwargs = {
                "vocab_size": opts.vocab_size,
                "embed_dim": opts.vocab_size,  # embedding & hidden hard-coded as the same-size as the vocab
                "hidden_size": opts.vocab_size,
                "max_len": opts.max_len,
                "cell": "lstm",
                "temperature": opts.gs_temperature,
            }
            rkwargs = {
                "vocab_size": opts.vocab_size,
                "embed_dim": opts.vocab_size,
                "hidden_size": opts.vocab_size,
                "cell": "lstm",
            }
        else:
            raise NotImplementedError(f"Unknown com channel : {opts.com_channel}")
        senders = [
            wrapper(
                Sender(
                    vision_module=find_module_from_name(vision_modules, module_name)[0],
                    input_dim=find_module_from_name(vision_modules, module_name)[1],
                    vocab_size=opts.vocab_size,
                    name=module_name,
                ),
                **kwargs,
            )
            for module_name in vision_model_names_senders
        ]
        receivers = [
            rwrapper(
                Receiver(
                    vision_module=find_module_from_name(vision_modules, module_name)[0],
                    input_dim=find_module_from_name(vision_modules, module_name)[1],
                    hidden_dim=opts.recv_hidden_dim,
                    output_dim=opts.vocab_size,
                    temperature=opts.recv_temperature,
                    name=module_name,
                ),
                **rkwargs,
            )
            for module_name in vision_model_names_receiver
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
    if not opts.com_channel == "lstm":
        _game = Game
    else:
        raise NotImplementedError("LSTM not implemented yet, replace GS by Reinforce")
        _game = SenderReceiverRnnGS  # BEWARE --> no baseline no opts channel

    game = _game(
        train_logging_strategy=train_logging_strategy,
        test_logging_strategy=test_logging_strategy,
        noisy=opts.noisy_channel,
        baseline="mean"
        if opts.com_channel == "reinforce" or opts.com_channel == "lstm"
        else None,
    )

    game = PopulationGame(
        game,
        agents_loss_sampler,
        aux_loss=opts.aux_loss,
        aux_loss_weight=opts.aux_loss_weight,
    )

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def build_second_game(opts):
    """
    used for retraining when adding a new agent to an existing population
    """

    pop_game = build_game(opts)
    load_from_checkpoint(pop_game, opts.base_checkpoint_path)

    for old_sender in pop_game.agents_loss_sampler.senders:
        for param in old_sender.parameters():
            param.requires_grad = False
    for old_receiver in pop_game.agents_loss_sampler.receivers:
        for param in old_receiver.parameters():
            param.requires_grad = False

    new_senders, new_receivers = build_senders_receivers(
        opts, opts.additional_senders, opts.additional_receivers
    )
    pop_game.agents_loss_sampler.avoid_training_old()
    pop_game.agents_loss_sampler.add_senders(new_senders)
    pop_game.agents_loss_sampler.add_receivers(new_receivers)

    # if opts.distributed_context.is_distributed:
    #     game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return pop_game
