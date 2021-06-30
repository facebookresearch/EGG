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

    def __getitem__(self, idx):
        """
        >>> b = Batch(torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([3]), {})
        >>> b[0]
        tensor([1.])
        >>> b[1]
        tensor([2.])
        >>> b[2]
        tensor([3.])
        >>> b[3]
        {}
        >>> b[6]
        Traceback (most recent call last):
            ...
        IndexError: Trying to access a wrong index in the batch
        """
        if idx == 0:
            return self.sender_input
        elif idx == 1:
            return self.labels
        elif idx == 2:
            return self.receiver_input
        elif idx == 3:
            return self.aux_input
        else:
            raise IndexError("Trying to access a wrong index in the batch")

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
        self.sender_input = move_to(self.sender_input, device)
        self.labels = move_to(self.labels, device)
        self.receiver_input = move_to(self.receiver_input, device)
        self.aux_input = move_to(self.aux_input, device)
        return self
