# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

class BaseEarlyStopper:
    """
    A base class, supports the running statistic which is could be used for early stopping
    """
    def __init__(self):
        self.train_stats = []
        self.validation_stats = []
        self.epoch = 0

    def should_stop(self) -> bool:
        raise NotImplementedError

    def update_values(self, validation_loss: float,
                      validation_rest: Dict[str, float],
                      train_loss: float,
                      train_rest: Dict[str, float],
                      epoch: int) -> bool:
        self.train_stats.append((train_loss.detach(), train_rest))
        self.validation_stats.append((validation_loss.detach(), validation_rest))
        self.epoch = epoch


class EarlyStopperAccuracy(BaseEarlyStopper):
    """
    Implements early stopping logic that stops training when a threshold on the validation loss
    is achieved.

    >>> early_stopper = EarlyStopperAccuracy(0.5)
    >>> epoch_stats = 100.0, {'acc': 1.0}, 100.0, {'acc': 0.2}  # generated within Trainer
    >>> early_stopper.update_values(*epoch_stats, epoch=1)
    >>> early_stopper.should_stop()
    True
    >>> epoch_stats = 100.0, {'acc': 0.0}, 100.0, {'acc': 0.2}  # generated within Trainer
    >>> early_stopper.update_values(*epoch_stats, epoch=2)
    >>> early_stopper.should_stop()
    False
    """
    def __init__(self, threshold: float) -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name 'acc'
        """
        super(EarlyStopperAccuracy, self).__init__()
        self.threshold = threshold

    def should_stop(self) -> bool:
        return self.validation_stats[-1][1]['acc'] > self.threshold
