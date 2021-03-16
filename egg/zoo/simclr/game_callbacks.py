# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()

        # TRAIN
        self.best_train_acc, self.best_train_loss, self.best_train_epoch = -float("inf"), float("inf"), -1
        self.last_train_acc, self.last_train_loss, self.last_train_epoch = 0., 0., 0
        # last_val_epoch useful for runs that end before the final epoch

        self.best_val_acc, self.best_val_loss, self.best_val_epoch = -float("inf"), float("inf"), -1
        self.last_val_acc, self.last_val_loss, self.last_val_epoch = 0., 0., 0
        # last_{train, val}_epoch useful for runs that end before the final epoch

    def on_epoch_end(self, loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = loss

        self.last_train_acc = logs.aux["acc"].mean().item()
        self.last_train_epoch = epoch
        self.last_train_loss = loss

    def on_test_end(self, loss, logs: Interaction, epoch: int):
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
            last_loss=self.last_train_loss
        )
        print(json.dumps(train_stats), flush=True)
        val_stats = dict(
            mode="best validation acc",
            best_epoch=self.best_val_epoch,
            best_acc=self.best_val_acc,
            best_loss=self.best_val_loss,
            last_epoch=self.last_val_epoch,
            last_acc=self.last_val_acc,
            last_loss=self.last_val_loss
        )
        print(json.dumps(val_stats), flush=True)


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""
    def __init__(
        self,
        shared: bool,
    ):
        super().__init__()

        self.shared = shared

    def save_vision_model(self, epoch=""):
        is_distributed = self.trainer.distributed_context.is_distributed
        rank = self.trainer.distributed_context.local_rank
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path and (
                    (not is_distributed) or (is_distributed and rank == 0)
                )
            ):
                if is_distributed:
                    # if distributed training the model is an instance of the DistributedDataParallel class
                    # and we need to unpack it from it.
                    vision_module = self.trainer.game.module.vision_module
                else:
                    vision_module = self.trainer.game.vision_module

                model_name = f"vision_module_{'shared' if self.shared else 'sender'}_{epoch if epoch else '_final'}.pt"
                torch.save(
                    vision_module.encoder.state_dict(),
                    self.trainer.checkpoint_path / model_name
                )

                if not self.shared:
                    model_name = f"vision_module_recv_{epoch if epoch else '_final'}.pt"
                    torch.save(
                        vision_module.encoder_recv.state_dict(),
                        self.trainer.checkpoint_path / model_name
                    )

    def on_train_end(self):
        self.save_vision_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        self.save_vision_model(epoch=epoch)


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_test_begin(self, epoch: int):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.validation_data.sampler.set_epoch(epoch)
