# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Optional, Dict, Union
from dataclasses import dataclass
from functools import cached_property
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
    aux: Dict[str, torch.Tensor]

    @cached_property
    def bsz(self):
        return self.sender_input.size(0)

    def to(self, *args, **kwargs):
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""
        def _to(x):
            if x is None or not torch.is_tensor(x): return x
            return x.to(*args, **kwargs)

        self.sender_input = _to(self.sender_input)
        self.receiver_input = _to(self.receiver_input)
        self.labels = _to(self.labels)
        self.message = _to(self.message)
        self.receiver_output = _to(self.receiver_output)
        self.message_length = _to(self.message_length)

        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

    def __add__(self, other: 'Interaction') -> 'Interaction':
        """
        >>> a = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> a.bsz
        1
        >>> b = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> c = a + b
        >>> c.bsz
        2
        >>> c
        Interaction(sender_input=tensor([1., 1.]), receiver_input=None, labels=None, message=tensor([1., 1.]), receiver_output=tensor([1., 1.]), message_length=None, aux={})
        >>> d = Interaction(torch.ones(1), torch.ones(1), None, torch.ones(1), torch.ones(1), None, {})
        >>> a + d # mishaped, should throw an exception
        Traceback (most recent call last):
        ...
        RuntimeError: Appending empty and non-empty interactions logs. Normally this shouldn't happen!
        """
        def _check_append(a, b):
            if a is None and b is None: return None
            if a is not None and b is not None:
                return torch.cat((a, b), dim=0)
            raise RuntimeError("Appending empty and non-empty interactions logs. "
                               "Normally this shouldn't happen!")

        aux = {}
        for k in self.aux:
            assert k in other.aux
            aux[k] = _check_append(self.aux[k], other.aux[k])

        return Interaction(
            sender_input=_check_append(self.sender_input, other.sender_input),
            receiver_input=_check_append(self.receiver_input, other.receiver_input),
            labels=_check_append(self.labels, other.labels),
            message=_check_append(self.message, other.message),
            message_length=_check_append(self.message_length, other.message_length),
            receiver_output=_check_append(self.receiver_output, other.receiver_output),
            aux=aux
        )
