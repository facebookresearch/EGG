# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .trainers import Trainer
from .callbacks import Callback, ConsoleLogger, TensorboardLogger, TemperatureUpdater, CheckpointSaver
from .util import init, get_opts, build_optimizer, dump_sender_receiver, move_to, get_summary_writer, close
from .early_stopping import EarlyStopperAccuracy
from .gs_wrappers import (GumbelSoftmaxWrapper,
                          SymbolGameGS, RelaxedEmbedding,
                          RnnSenderGS, RnnReceiverGS,
                          SenderReceiverRnnGS, SymbolReceiverWrapper)

from .reinforce_wrappers import (ReinforceWrapper, SymbolGameReinforce,
                                 ReinforceDeterministicWrapper, RnnReceiverReinforce,
                                 RnnSenderReinforce, SenderReceiverRnnReinforce,
                                 RnnReceiverDeterministic, TransformerReceiverDeterministic,
                                 TransformerSenderReinforce)

from .rnn import RnnEncoder

__all__ = [
    'Trainer',
    'get_opts',
    'init',
    'build_optimizer',
    'Callback',
    'EarlyStopperAccuracy',
    'ConsoleLogger',
    'TensorboardLogger',
    'TemperatureUpdater',
    'CheckpointSaver',
    'ReinforceWrapper',
    'GumbelSoftmaxWrapper',
    'SymbolGameGS',
    'SymbolGameReinforce',
    'ReinforceDeterministicWrapper',
    'RelaxedEmbedding',
    'RnnReceiverReinforce',
    'RnnSenderReinforce',
    'SenderReceiverRnnReinforce',
    'RnnReceiverDeterministic',
    'RnnSenderGS',
    'RnnReceiverGS',
    'SenderReceiverRnnGS',
    'dump_sender_receiver',
    'move_to',
    'get_summary_writer',
    'close',
    'SymbolReceiverWrapper',
    'TransformerReceiverDeterministic',
    'TransformerSenderReinforce',
    'RnnEncoder'
]
