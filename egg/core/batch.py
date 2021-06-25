# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import torch

from egg.core.util import move_to


class Batch:
    def __init__(
        self,
        sender_input: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[Dict[Any, Any]] = None,
    ):
        self.sender_input = sender_input
        self.labels = labels
        self.receiver_input = receiver_input
        self.aux_input = aux_input

    def __iter__(self):
        """
        >>> _ = torch.manual_seed(111)
        >>> sender_input = torch.rand(2, 2)
        >>> labels = torch.rand(2, 2)
        >>> batch = Batch(sender_input, labels)
        >>> it = batch.__iter__()
        >>> it_sender_input = next(it)
        >>> torch.allclose(sender_input, it_sender_input)
        True
        >>> it_labels = next(it)
        >>> torch.allclose(labels, it_labels)
        True
        """
        return iter(
            [self.sender_input, self.labels, self.receiver_input, self.aux_input]
        )

    def to(self, device: torch.device):
        """Method to move all (nested) tensors of the batch to a specific device.
        This operation doest not change the original batch element and returns a new Batch instance.
        """
        moved_batch = move_to(
            [self.sender_input, self.labels, self.receiver_input, self.aux], device
        )
        return Batch(*moved_batch)
