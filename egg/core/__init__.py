# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .callbacks import (
    Callback,
    CheckpointSaver,
    ConsoleLogger,
    InteractionSaver,
    ProgressBarLogger,
    TemperatureUpdater,
    TensorboardLogger,
)
from .continous_communication import (
    ContinuousLinearReceiver,
    ContinuousLinearSender,
    SenderReceiverContinuousCommunication,
)
from .early_stopping import EarlyStopperAccuracy
from .gs_wrappers import (
    GumbelSoftmaxWrapper,
    RelaxedEmbedding,
    RnnReceiverGS,
    RnnSenderGS,
    SenderReceiverRnnGS,
    SymbolGameGS,
    SymbolReceiverWrapper,
)
from .interaction import Interaction, LoggingStrategy, dump_interactions
from .language_analysis import (
    Disent,
    MessageEntropy,
    PrintValidationEvents,
    TopographicSimilarity,
)
from .losses import DiscriminationLoss, NTXentLoss, ReconstructionLoss
from .population import FullSweepAgentSampler, PopulationGame, UniformAgentSampler
from .reinforce_wrappers import (
    CommunicationRnnReinforce,
    ReinforceDeterministicWrapper,
    ReinforceWrapper,
    RnnReceiverDeterministic,
    RnnReceiverReinforce,
    RnnSenderReinforce,
    SenderReceiverRnnReinforce,
    SymbolGameReinforce,
    TransformerReceiverDeterministic,
    TransformerSenderReinforce,
)
from .rnn import RnnEncoder
from .trainers import Trainer
from .util import (
    build_optimizer,
    close,
    find_lengths,
    get_opts,
    get_summary_writer,
    init,
    move_to,
)

__all__ = [
    "Trainer",
    "get_opts",
    "init",
    "build_optimizer",
    "Callback",
    "EarlyStopperAccuracy",
    "ConsoleLogger",
    "ProgressBarLogger",
    "TensorboardLogger",
    "TemperatureUpdater",
    "InteractionSaver",
    "CheckpointSaver",
    "ReinforceWrapper",
    "GumbelSoftmaxWrapper",
    "SymbolGameGS",
    "SymbolGameReinforce",
    "ReinforceDeterministicWrapper",
    "RelaxedEmbedding",
    "RnnReceiverReinforce",
    "RnnSenderReinforce",
    "SenderReceiverRnnReinforce",
    "CommunicationRnnReinforce",
    "RnnReceiverDeterministic",
    "RnnSenderGS",
    "RnnReceiverGS",
    "SenderReceiverRnnGS",
    "dump_interactions",
    "move_to",
    "get_summary_writer",
    "close",
    "SymbolReceiverWrapper",
    "TransformerReceiverDeterministic",
    "TransformerSenderReinforce",
    "RnnEncoder",
    "find_lengths",
    "LoggingStrategy",
    "Interaction",
    "MessageEntropy",
    "TopographicSimilarity",
    "Disent",
    "PrintValidationEvents",
    "DiscriminationLoss",
    "ReconstructionLoss",
    "NTXentLoss",
    "UniformAgentSampler",
    "FullSweepAgentSampler",
    "PopulationGame",
    "ContinuousLinearSender",
    "ContinuousLinearReceiver",
    "SenderReceiverContinuousCommunication",
]
