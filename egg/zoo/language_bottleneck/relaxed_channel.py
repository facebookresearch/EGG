# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.distributions import RelaxedOneHotCategorical


class AlwaysRelaxedWrapper(nn.Module):
    def __init__(self, agent, temperature=1.0):
        super().__init__()
        self.agent = agent
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        if self.training:
            return RelaxedOneHotCategorical(logits=logits, temperature=self.temperature).rsample()
        else:
            return (logits / self.temperature).softmax(dim=1)
