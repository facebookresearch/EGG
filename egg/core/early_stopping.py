# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any, List, Tuple

from .callbacks import Callback


class BaseEarlyStopper(Callback):
    """
    A base class, supports the running statistic which is could be used for early stopping
    """
    def __init__(self):
        super(BaseEarlyStopper, self).__init__()
        self.train_stats: List[Tuple[float, Dict[str, Any]]] = []
        self.validation_stats: List[Tuple[float, Dict[str, Any]]] = []
        self.epoch: int = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None) -> None:
        self.epoch += 1
        self.train_stats.append((loss, logs))

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None) -> None:
        self.validation_stats.append((loss, logs))
        self.trainer.should_stop = self.should_stop()

    def should_stop(self) -> bool:
        raise NotImplementedError()


class EarlyStopperAccuracy(BaseEarlyStopper):
    """
    Implements early stopping logic that stops training when a threshold on a metric
    is achieved.
    """
    def __init__(self, threshold: float, field_name: str = 'acc') -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name `field_name`)
        :param field_name: the name of the metric return by loss function which should be evaluated against stopping
            criterion (default: "acc")
        """
        super(EarlyStopperAccuracy, self).__init__()
        self.threshold = threshold
        self.field_name = field_name

    def should_stop(self) -> bool:
        assert self.trainer.validation_data is not None, 'Validation data must be provided for early stooping to work'
        return self.validation_stats[-1][1][self.field_name] > self.threshold
