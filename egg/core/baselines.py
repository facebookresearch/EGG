# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod, ABC
from collections import defaultdict

import torch


class Baseline(ABC):
    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        """Update internal state according to the observed loss
            loss (torch.Tensor): batch of losses
        """
        pass

    @abstractmethod
    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        """Return baseline for the loss
        Args:
            loss (torch.Tensor): batch of losses be baselined
        """
        pass


class NoBaseline(Baseline):
    """Baseline that does nothing (constant zero baseline)"""

    def __init__(self):
        super().__init__()

    def update(self, loss: torch.Tensor) -> None:
        pass

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=loss.device)


class MeanBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (loss.detach().mean().item() -
                               self.mean_baseline) / self.n_points

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline


class BuiltInBaseline(Baseline):
    """Built-in baseline; for any row in the batch, the mean of all other rows serves as a control variate.
    To use BuiltInBaseline, rows in the batch must be independent. Most likely BuiltInBaseline 
    would work poorly for small batch sizes.
    """

    def __init__(self):
        super().__init__()

    def update(self, _: torch.Tensor) -> None:
        pass

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if len(loss.size()) == 0 or loss.size(0) <= 1:
            return loss
        bsz = loss.size(0)

        loss_detached = loss.detach()
        mean = loss_detached.mean()

        baseline = (mean * bsz - loss_detached) / (bsz - 1.0)

        return baseline
