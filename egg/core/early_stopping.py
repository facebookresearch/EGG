# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any, List, Tuple

from .callbacks import Callback
from .interaction import Interaction


class BaseEarlyStopper(Callback):
    """
    A base class, supports the running statistic which is could be used for early stopping
    """
    def __init__(self, validation: bool = True):
        super(BaseEarlyStopper, self).__init__()
        self.train_stats: List[Tuple[float, Interaction]] = []
        self.validation_stats: List[Tuple[float, Interaction]] = []
        self.epoch: int = 0
        self.validation = validation

    def on_epoch_end(self, loss: float, logs: List[Interaction]) -> None:
        if self.validation:
            return
        self.epoch += 1
        self.train_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop()

    def on_test_end(self, loss: float, logs: List[Interaction]) -> None:
        if not self.validation:
            return
        self.validation_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop()

    def should_stop(self) -> bool:
        raise NotImplementedError()


class EarlyStopperAccuracy(BaseEarlyStopper):
    """
    Implements early stopping logic that stops training when a threshold on a metric
    is achieved.
    """
    def __init__(self, threshold: float, field_name: str = 'acc', validation: bool = True) -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name `field_name`)
        :param field_name: the name of the metric return by loss function which should be evaluated against stopping
            criterion (default: "acc")
        :param validation: whether the statistics on the validation (or training, if False) data should be checked
        """
        super(EarlyStopperAccuracy, self).__init__(validation)
        self.threshold = threshold
        self.field_name = field_name

    def should_stop(self) -> bool:
        if self.validation:
            assert self.validation_stats, 'Validation data must be provided for early stooping to work'
            loss, last_epoch_interactions = self.validation_stats[-1]
        else:
            assert self.train_stats, 'Training data must be provided for early stooping to work'
            loss, last_epoch_interactions = self.train_stats[-1]

        metric_sum = sum(x.aux[self.field_name].sum() for x in last_epoch_interactions)
        normalizer = sum(x.bsz for x in last_epoch_interactions)

        return metric_sum / normalizer >= self.threshold
