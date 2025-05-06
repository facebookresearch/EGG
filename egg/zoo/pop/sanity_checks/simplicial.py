# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
from torch import nn
from torch.nn import functional as F

from xformers import _is_triton_available

Self = TypeVar("Self", bound="SimplicialEmbedding")

@dataclass
class SimplicialEmbeddingConfig:
    L: int
    temperature: float


class SimplicialEmbedding(torch.nn.Module):
    """
    An implementation of the "Simplicial Embeddings"_, as proposed by Lavoie et. al

    Arguments:
        - L: the number of embedding chunks
        - temperature: optional scaling parameter for the softmax operation.
            A small (<1.) temperature will lead to a sparse representation (up to one-hot),
            while a large (>1.) temperature will make the vector more uniform

    _"Simplicial Embeddings": https://arxiv.org/pdf/2204.00616.pdf
    """

    def __init__(self, L: int, temperature: Optional[float] = None) -> None:
        super().__init__()
        self.L = L
        self.temperature = temperature
        self.printing=True


    def forward(self, x: torch.Tensor, aux_input=None) -> torch.Tensor:
        assert (
            x.shape[-1] % self.L == 0
        ), f"The embedding dimension {x.shape[-1]} is not divisible by the chosen L parameter {self.L}"

        # Seperate the input tensor into V chunks
        B, E = x.shape
        V = E // self.L
        # print("original ", x)
        Vs = x.reshape(B, self.L, V)
        if self.printing:
            #get shape of simplicial embedding
            print("original ", x.shape)
            print("Vs shape ", Vs.shape)
            self.printing=False
        # print("Vs", Vs)
        # Softmax normalize them, with the proposed temperature
        # This is done over the last dimension, so only within Vs
        if self.temperature is not None:
            Vs /= self.temperature
        # print("Vs/temp ", Vs)
        if False:#_is_triton_available():
            from xformers.triton.softmax import softmax as triton_softmax

            Vs = triton_softmax(
                Vs, mask=None, causal=False
            )  # the softmax is on the last dimension
        else:
            Vs = torch.nn.functional.softmax(Vs, dim=-1)
        # Concatenate back and return
        # print("normalized ", Vs)
        return Vs.reshape(B, E)

    @classmethod
    def from_config(cls: Type[Self], config: SimplicialEmbeddingConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        return cls(**fields)


class SimplicialWrapper(torch.nn.Module):
    """
    Use simplicial embedding as a trainable wrapper around a pretrained model
    """

    def __init__(self, vision_module, L: int, temperature: Optional[float] = None, hidden_size=None, v_output_dim=None) -> None:
        super().__init__()
        self.vision_module = vision_module
        self.L = L
        self.temperature = temperature
        self.simplicial = SimplicialEmbedding(L, temperature)
        # add optional linear layer
        if hidden_size is not None:
            self.linear = nn.Linear(v_output_dim, hidden_size)
    def forward(self, x: torch.Tensor, aux_input=None) -> torch.Tensor:
        x = self.vision_module(x)
        if hasattr(self, "linear"):
            x = self.linear(x)
        # print("output ", x)
        x = self.simplicial(x)
        return x

class Empty_wrapper(torch.nn.Module):
    def __init__(self, receiver):
        super().__init__()
        self.receiver = receiver

    def forward(self, x: torch.Tensor, receiver_input, aux_input=None) -> torch.Tensor:
        return self.receiver(
            x,
            receiver_input,
            aux_input,
        )
