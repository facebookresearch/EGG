# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from egg.core import Callback, ConsoleLogger


def get_callbacks(is_distributed: bool = False) -> List[Callback]:
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
    ]

    return callbacks
