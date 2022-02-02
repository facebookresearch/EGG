# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
import torch.nn as nn

from egg.core import ConsoleLogger, TemperatureUpdater
from egg.zoo.emcom_as_ssl.game_callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter,
)


def get_callbacks(
    shared_vision: bool,
    n_epochs: int,
    checkpoint_dir: str,
    senders: nn.Module,
    train_gs_temperature: bool = False,
    minimum_gs_temperature: float = 0.1,
    update_gs_temp_frequency: int = 1,
    gs_temperature_decay: float = 1.0,
    is_distributed: bool = False,
):
    # Mat : As in em_as_ssl except with multi agent support for gs temperature update
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
    ]

    if is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())
    for sender in senders:
        if hasattr(sender, "temperature") and (not train_gs_temperature):
            callbacks.append(
                TemperatureUpdater(
                    sender,
                    minimum=minimum_gs_temperature,
                    update_frequency=update_gs_temp_frequency,
                    decay=gs_temperature_decay,
                )
            )

    return callbacks
