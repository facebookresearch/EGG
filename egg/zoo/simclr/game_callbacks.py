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
        self.best_train_acc, self.best_train_loss, self.best_train_epoch = (
            -float("inf"),
            float("inf"),
            -1,
        )
        self.last_train_acc, self.last_train_loss, self.last_train_epoch = 0.0, 0.0, 0
        # last_val_epoch useful for runs that end before the final epoch

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_train_acc:
            self.best_train_acc = logs.aux["acc"].mean().item()
            self.best_train_epoch = epoch
            self.best_train_loss = _loss

        self.last_train_acc = logs.aux["acc"].mean().item()
        self.last_train_epoch = epoch
        self.last_train_loss = _loss

    def on_train_end(self):
        is_distributed = self.trainer.distributed_context.is_distributed
        is_leader = self.trainer.distributed_context.is_leader
        if (not is_distributed) or (is_distributed and is_leader):
            train_stats = dict(
                mode="train",
                epoch=self.best_train_epoch,
                acc=self.best_train_acc,
                loss=self.best_train_loss,
            )
            print(json.dumps(train_stats), flush=True)


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""

    def __init__(self):
        super().__init__()

    def save_vision_model(self, epoch=""):
        is_distributed = self.trainer.distributed_context.is_distributed
        is_leader = self.trainer.distributed_context.is_leader
        if hasattr(self.trainer, "checkpoint_path"):
            if self.trainer.checkpoint_path and (
                (not is_distributed) or (is_distributed and is_leader)
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                if is_distributed:
                    # if distributed training the model is an instance of
                    # DistributedDataParallel and we need to unpack it from it.
                    vision_module = self.trainer.game.module.vision_module
                else:
                    vision_module = self.trainer.game.vision_module
                torch.save(
                    vision_module.encoder.state_dict(),
                    self.trainer.checkpoint_path
                    / f"vision_module{epoch if epoch else '_final'}.pt",
                )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.save_vision_model(epoch=epoch)

    def on_train_end(self):
        self.save_vision_model()


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""

    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch):
        # just being cautious here given that non distributed jobs won't have probaly have distributed sampler set
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)
