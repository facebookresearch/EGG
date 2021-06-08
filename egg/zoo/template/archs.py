# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

from egg.core.interaction import Interaction


class Sender(nn.Module):
    def __init__(self):
        super(Sender, self).__init__()

    def forward(self, sender_input: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor, Any]]:
        pass


class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()

    def forward(
        self,
        message: torch.Tensor,
        receiver_input: torch.Tensor = None
    ) -> Union[torch.Tensor, List[torch.Tensor, Any]]:
        pass


class Game(nn.Module):
    def __init__(self):
        super(Game, self).__init__()

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Interaction]:
        pass
