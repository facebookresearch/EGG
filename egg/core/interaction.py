# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as distrib


class LoggingStrategy:
    """
   Logic applied to log interaction between receiver and sender.
   Each interaction is logged at each step (batch) of an epoch.
   Args:
       store_sender_input: if to store the sender input
       store_receiver_input: if to store the receiver input
       store_labels: if to store the labels
       store_message: if to store the message
       store_receiver_output: if to store the receiver output
       store_message_length: if to store the message length
   """

    def __init__(
            self,
            store_sender_input: bool = True,
            store_receiver_input: bool = True,
            store_labels: bool = True,
            store_message: bool = True,
            store_receiver_output: bool = True,
            store_message_length: bool = True,
    ):
        self.store_sender_input = store_sender_input
        self.store_receiver_input = store_receiver_input
        self.store_labels = store_labels
        self.store_message = store_message
        self.store_receiver_output = store_receiver_output
        self.store_message_length = store_message_length

    def filtered_interaction(
            self,
            sender_input: Optional[torch.Tensor],
            receiver_input: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
            message: Optional[torch.Tensor],
            receiver_output: Optional[torch.Tensor],
            message_length: Optional[torch.Tensor],
            aux: Dict[str, torch.Tensor],
    ):
        return Interaction(
            sender_input=sender_input if self.store_sender_input else None,
            receiver_input=receiver_input if self.store_receiver_input else None,
            labels=labels if self.store_labels else None,
            message=message if self.store_message else None,
            receiver_output=receiver_output if self.store_receiver_output else None,
            message_length=message_length if self.store_message_length else None,
            aux=aux,
        )

    @classmethod
    def minimal(cls):
        args = [False] * 5 + [True]
        return cls(*args)

    @classmethod
    def maximal(cls):
        return cls()


class RandomLogging(LoggingStrategy):
    """
    Log strategy based on random probability
    """

    def __init__(self, store_prob=1, random_seed=42, *args):

        super().__init__(*args)
        random.seed(a=random_seed)

    def filtered_interaction(
            self,
            sender_input: Optional[torch.Tensor],
            receiver_input: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
            message: Optional[torch.Tensor],
            receiver_output: Optional[torch.Tensor],
            message_length: Optional[torch.Tensor],
            aux: Dict[str, torch.Tensor],
    ):
        rnd = random.random()
        should_store = rnd < self.store_prob
        return Interaction(
            sender_input=sender_input if should_store else None,
            receiver_input=receiver_input if should_store else None,
            labels=labels if should_store else None,
            message=message if should_store else None,
            receiver_output=receiver_output if should_store else None,
            message_length=message_length if should_store else None,
            aux=aux if should_store else None,
        )


@dataclass
class Interaction:
    # incoming data
    sender_input: Optional[torch.Tensor]
    receiver_input: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]

    # what agents produce
    message: Optional[torch.Tensor]
    receiver_output: Optional[torch.Tensor]

    # auxilary info
    message_length: Optional[torch.Tensor]
    aux: Dict[str, torch.Tensor]

    @property
    def size(self):
        interaction_fields = [
            self.sender_input,
            self.receiver_input,
            self.labels,
            self.message,
            self.receiver_output,
            self.message_length,
        ]
        for t in interaction_fields:
            if t is not None:
                return t.size(0)
        raise RuntimeError("Cannot determine interaction log size; it is empty.")

    def to(self, *args, **kwargs) -> "Interaction":
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""

        def _to(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(*args, **kwargs)

        self.sender_input = _to(self.sender_input)
        self.receiver_input = _to(self.receiver_input)
        self.labels = _to(self.labels)
        self.message = _to(self.message)
        self.receiver_output = _to(self.receiver_output)
        self.message_length = _to(self.message_length)

        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self

    @staticmethod
    def from_iterable(interactions: Iterable["Interaction"]) -> "Interaction":
        """
        >>> a = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> a.size
        1
        >>> b = Interaction(torch.ones(1), None, None, torch.ones(1), torch.ones(1), None, {})
        >>> c = Interaction.from_iterable((a, b))
        >>> c.size
        2
        >>> c
        Interaction(sender_input=tensor([1., 1.]), receiver_input=None, labels=None, message=tensor([1., 1.]), receiver_output=tensor([1., 1.]), message_length=None, aux={})
        """

        def _check_cat(lst):
            if all(x is None for x in lst):
                return None
            # if some but not all are None: filter out None
            lst = [x for x in lst if x is not None]
            return torch.cat(lst, dim=0)

        assert interactions, "list must not be empty"
        assert all(len(x.aux) == len(interactions[0].aux) for x in interactions)

        aux = {}
        for k in interactions[0].aux:
            aux[k] = _check_cat([x.aux[k] for x in interactions])

        return Interaction(
            sender_input=_check_cat([x.sender_input for x in interactions]),
            receiver_input=_check_cat([x.receiver_input for x in interactions]),
            labels=_check_cat([x.labels for x in interactions]),
            message=_check_cat([x.message for x in interactions]),
            message_length=_check_cat([x.message_length for x in interactions]),
            receiver_output=_check_cat([x.receiver_output for x in interactions]),
            aux=aux,
        )

    @staticmethod
    def empty() -> "Interaction":
        return Interaction(None, None, None, None, None, None, {})

    @staticmethod
    def gather_distributed_interactions(log: "Interaction") -> Optional["Interaction"]:
        assert (
            distrib.is_initialized()
        ), "torch.distributed must be initialized beforehand"
        world_size = distrib.get_world_size()

        def send_collect_tensor(tnsr):
            assert tnsr is not None

            tnsr = tnsr.contiguous().cuda()
            lst = [torch.zeros_like(tnsr) for _ in range(world_size)]
            distrib.all_gather(lst, tnsr)
            return torch.cat(lst, dim=0).to("cpu")

        def send_collect_dict(d):
            new_d = {}

            for k, v in d.items():
                if v is not None:
                    v = send_collect_tensor(v)
                new_d[k] = v

            return new_d

        interaction_as_dict = dict(
            (name, getattr(log, name))
            for name in [
                "sender_input",
                "receiver_input",
                "labels",
                "message",
                "message_length",
                "receiver_output",
            ]
        )

        interaction_as_dict = send_collect_dict(interaction_as_dict)
        synced_aux = send_collect_dict(log.aux)
        interaction_as_dict["aux"] = synced_aux
        synced_interacton = Interaction(**interaction_as_dict)
        assert log.size * world_size == synced_interacton.size
        return synced_interacton
