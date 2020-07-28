# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional, Dict, Union
from dataclasses import dataclass
import torch

@dataclass
class Interaction:
    # incoming data
    sender_input: torch.Tensor
    receiver_input: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]

    # what agents produce
    message: torch.Tensor
    receiver_output: torch.Tensor

    # auxilary info
    message_length: Optional[torch.Tensor]
    aux: Dict[str, Union[float, int, torch.Tensor]]


    @staticmethod
    def from_batch(sender_input, receiver_input, labels):
        interacton = Interaction(
            sender_input,
            receiver_input,
            labels,
            message=None,
            receiver_output=None,
            message_length=None,
            aux={}
        )

