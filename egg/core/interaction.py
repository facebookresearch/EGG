# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import warnings
from dataclasses import dataclass
from itertools import chain, groupby
from typing import Dict, Iterable, Optional, Union

import torch
import torch.distributed as distrib

from egg.core.batch import Batch


@dataclass(repr=True, unsafe_hash=True)
class Interaction:
    # incoming data
    sender_input: Optional[Union[torch.Tensor, list]]
    receiver_input: Optional[Union[torch.Tensor, list]]
    labels: Optional[Union[torch.Tensor, list]]
    aux_input: Optional[Dict[str, Union[torch.Tensor, list]]]

    # what agents produce
    message: Optional[Union[torch.Tensor, list]]
    receiver_output: Optional[Union[torch.Tensor, list]]

    # auxilary info
    message_length: Optional[Union[torch.Tensor, list]]
    aux: Dict[str, Union[torch.Tensor, list]]

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

    def is_empty(self) -> bool:
        return self == self.empty()

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

        if self.aux_input:
            self.aux_input = dict((k, _to(v)) for k, v in self.aux_input.items())
        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self

    @staticmethod
    def from_iterable(interactions: Iterable["Interaction"]) -> "Interaction":
        """
        >>> a = Interaction(torch.ones(1), None, None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> a.size
        1
        >>> b = Interaction(torch.ones(1), None, None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> c = Interaction.from_iterable((a, b))
        >>> c.size
        2
        >>> c
        Interaction(sender_input=tensor([[1.],
                [1.]]), receiver_input=None, labels=None, aux_input={}, message=tensor([[1.],
                [1.]]), receiver_output=tensor([[1.],
                [1.]]), message_length=None, aux={})
        >>> d = Interaction(torch.ones(1), torch.ones(1), None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> _ = Interaction.from_iterable((a, d)) # mishaped, should throw an exception
        Traceback (most recent call last):
        ...
        RuntimeError: Appending empty and non-empty interactions logs. Normally this shouldn't happen!
        """

        def _all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        def _check_cat(lst):
            if all(x is None for x in lst):
                return None
            # if some but not all are None: not good
            if any(x is None for x in lst):
                raise RuntimeError(
                    "Appending empty and non-empty interactions logs. "
                    "Normally this shouldn't happen!"
                )

            if all([isinstance(x, torch.Tensor) for x in lst]):
                return torch.stack(lst, dim=0)
            elif all([isinstance(x, list) for x in lst]):
                # lst is a list of lists
                if _all_equal([len(x) for x in lst]):
                    return lst
                else:
                    raise RuntimeError(
                        "The element of the interaction logs must have the same size!"
                    )
            elif all([isinstance(x, type(lst[0])) for x in lst]):
                # if not a list or tensor then return as long as everything has the same type
                return lst
            else:
                raise RuntimeError(
                    "Element of interaction have different datatypes, be sure to choose either list or torch.Tensor"
                )

        # filter out empty interactions
        interactions = [x for x in interactions if not x.is_empty()]

        if len(interactions) == 0:
            # we still need at least one empty interaction in the list, so it is empty after the filtering add one
            interactions = [Interaction.empty()]

        has_aux_input = interactions[0].aux_input is not None

        for x in interactions:
            assert len(x.aux) == len(interactions[0].aux)
            if has_aux_input:
                assert len(x.aux_input) == len(
                    interactions[0].aux_input
                ), "found two interactions of different aux_info size"
            else:
                assert (
                    not x.aux_input
                ), "some aux_info are defined some are not, this should not happen"

        aux_input = None
        if has_aux_input:
            aux_input = {}
            for k in interactions[0].aux_input:
                aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
        aux = {}
        for k in interactions[0].aux:
            aux[k] = _check_cat([x.aux[k] for x in interactions])

        return Interaction(
            sender_input=_check_cat([x.sender_input for x in interactions]),
            receiver_input=_check_cat([x.receiver_input for x in interactions]),
            labels=_check_cat([x.labels for x in interactions]),
            aux_input=aux_input,
            message=_check_cat([x.message for x in interactions]),
            message_length=_check_cat([x.message_length for x in interactions]),
            receiver_output=_check_cat([x.receiver_output for x in interactions]),
            aux=aux,
        )

    @staticmethod
    def empty() -> "Interaction":
        return Interaction(None, None, None, {}, None, None, None, {})

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
            if not d or d is None:
                return {}

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

        synced_aux_input = send_collect_dict(log.aux_input)
        interaction_as_dict["aux_input"] = synced_aux_input
        synced_aux = send_collect_dict(log.aux)
        interaction_as_dict["aux"] = synced_aux

        synced_interacton = Interaction(**interaction_as_dict)

        assert log.size * world_size == synced_interacton.size
        return synced_interacton


def old_signature_warning(func):
    """This is a decorator which is used to warn users on the new signature for the 'filtered_interaction'.
    ."""

    def warn():
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            f"Call to deprecated version of '{func.__name__}'. You should pass an Interaction class based on the input dict ",
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if "batch_id" not in kwargs.keys():
            warn()
            interaction = Interaction(**kwargs)
            return func(*args, interaction=interaction, batch_id=-1)
        return func(*args, **kwargs)

    return new_func


@dataclass(repr=True, eq=True)
class LoggingStrategy:
    store_sender_input: bool = True
    store_receiver_input: bool = True
    store_labels: bool = True
    store_aux_input: bool = True
    store_message: bool = True
    store_receiver_output: bool = True
    store_message_length: bool = True
    logging_step: int = 0

    @old_signature_warning
    def filtered_interaction(self, interaction: Interaction, batch_id: int):

        # when logging_step==0 log. Else consider batch_id value
        if self.logging_step == 0 or batch_id % self.logging_step == 0:

            filtered_interaction = Interaction(
                sender_input=interaction.sender_input
                if self.store_sender_input
                else None,
                receiver_input=interaction.receiver_input
                if self.store_receiver_input
                else None,
                labels=interaction.labels if self.store_labels else None,
                aux_input=interaction.aux_input if self.store_aux_input else None,
                message=interaction.message if self.store_message else None,
                receiver_output=interaction.receiver_output
                if self.store_receiver_output
                else None,
                message_length=interaction.message_length
                if self.store_message_length
                else None,
                aux=interaction.aux,
            )
        else:
            filtered_interaction = Interaction.empty()

        return filtered_interaction

    @classmethod
    def minimal(cls):
        args = [False] * 6 + [True]
        return cls(*args)

    @classmethod
    def maximal(cls):
        return cls()


def dump_interactions(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    variable_length: bool,
    device: Optional[torch.device] = None,
    apply_padding: bool = True,
) -> Interaction:
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether the messages should be argmaxed over the last dimension.
        Handy, if Gumbel-Softmax relaxation was used for training.
    :param variable_length: whether variable-length communication is used.
    :param device: device (e.g. 'cuda') to be used.
    :return: The entire log of agent interactions, represented as an Interaction instance.
    """
    train_state = game.training  # persist so we restore it back
    game.eval()
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    full_interaction = None

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(device)
            _, interaction = game(*batch)
            interaction = interaction.to("cpu")

            if gs:
                interaction.message = interaction.message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded
            if apply_padding and variable_length:
                assert interaction.message_length is not None
                for i in range(interaction.size):
                    length = interaction.message_length[i].long().item()
                    interaction.message[i, length:] = 0  # 0 is always EOS

            full_interaction = (
                full_interaction + interaction
                if full_interaction is not None
                else interaction
            )

    game.train(mode=train_state)
    return interaction
