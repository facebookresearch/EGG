# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()

        # TRAIN
        self.best_train_acc, self.best_train_loss, self.best_train_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_train_acc, self.last_train_loss, self.last_train_epoch = 0.0, 0.0, 0
        # last_val_epoch useful for runs that end before the final epoch

        self.best_val_acc, self.best_val_loss, self.best_val_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_val_acc, self.last_val_loss, self.last_val_epoch = 0.0, 0.0, 0
        # last_{train, val}_epoch useful for runs that end before the final epoch

    def on_epoch_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = loss

        self.last_train_acc = logs.aux["acc"].mean().item()
        self.last_train_epoch = epoch
        self.last_train_loss = loss

    def on_validation_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_val_acc:
            self.best_val_acc = logs.aux["acc"].mean().item()
            self.best_val_epoch = epoch
            self.best_val_loss = loss

        self.last_val_acc = logs.aux["acc"].mean().item()
        self.last_val_epoch = epoch
        self.last_val_loss = loss

    def on_train_end(self):
        train_stats = dict(
            mode="best train acc",
            best_epoch=self.best_train_epoch,
            best_acc=self.best_train_acc,
            best_loss=self.best_train_loss,
            last_epoch=self.last_train_epoch,
            last_acc=self.last_train_acc,
            last_loss=self.last_train_loss,
        )
        print(json.dumps(train_stats), flush=True)
        val_stats = dict(
            mode="best validation acc",
            best_epoch=self.best_val_epoch,
            best_acc=self.best_val_acc,
            best_loss=self.best_val_loss,
            last_epoch=self.last_val_epoch,
            last_acc=self.last_val_acc,
            last_loss=self.last_val_loss,
        )
        print(json.dumps(val_stats), flush=True)


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_validation_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)


class ConsoleLogger(Callback):
    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)
