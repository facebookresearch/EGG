# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

from egg.core.util import move_to


@dataclass(repr=True, unsafe_hash=True)
class Batch:
    sender_input: torch.Tensor
    labels: Optional[torch.Tensor] = None
    receiver_input: Optional[torch.Tensor] = None
    aux: Optional[Dict[Any, Any]] = None

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
        return iter([self.sender_input, self.labels, self.receiver_input, self.aux])


def generate_batch_from_raw(raw_batch: Iterable) -> Batch:
    """
    >>> _ = torch.manual_seed(111)
    >>> raw_batch = (torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2), None)
    >>> generate_batch_from_raw(raw_batch)
    Batch(sender_input=tensor([[0.7156, 0.9140],
            [0.2819, 0.2581]]), labels=tensor([[0.6311, 0.6001],
            [0.9312, 0.2153]]), receiver_input=tensor([[0.6033, 0.7328],
            [0.1857, 0.5101]]), aux=None)
    >>> raw_batch = (torch.rand(2, 2))
    >>> generate_batch_from_raw(raw_batch)
    Batch(sender_input=tensor([[0.7545, 0.2884],
            [0.5775, 0.0358]]), labels=None, receiver_input=None, aux=None)
    """
    if isinstance(raw_batch, torch.Tensor):
        return Batch(sender_input=raw_batch)
    return Batch(*raw_batch)


def generate_identity_batch(raw_batch: Batch) -> Batch:
    return raw_batch


def move_batch_to(batch: Batch, device: torch.device) -> Batch:
    """Method to move all (nested) tensors of the batch to a specific device.
    This operation doest not change the original batch element and returns a new Batch instance.
    """
    moved_batch = move_to(
        [batch.sender_input, batch.labels, batch.receiver_input, batch.aux], device
    )
    return Batch(*moved_batch)
