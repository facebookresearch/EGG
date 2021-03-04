# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()

        self.best_acc = -1.
        self.best_loss = float("inf")
        self.best_epoch = -1

        self.last_acc = 0.
        self.last_loss = 0.
        self.last_epoch = 0

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_acc:
            self.best_acc = logs.aux["acc"].mean().item()
            self.best_epoch = epoch
            self.best_loss = _loss

            self.last_acc = logs.aux["acc"].mean().item()
            self.last_epoch = epoch
            self.last_loss = _loss

    def on_train_end(self):
        is_distributed = self.trainer.distributed_context.is_distributed
        rank = self.trainer.distributed_context.local_rank
        if (not is_distributed) or (is_distributed and rank == 0):
            print(f"BEST: epoch {self.best_epoch}, acc: {self.best_acc}, loss: {self.best_loss}")
            print(f"LAST: epoch {self.last_epoch}, acc: {self.last_acc}, loss: {self.last_loss}")


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""
    def __init__(
        self,
        shared: bool,
    ):
        super().__init__()

        self.shared = shared

    def on_train_end(self):
        is_distributed = self.trainer.distributed_context.is_distributed
        rank = self.trainer.distributed_context.local_rank
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path and (
                    (not is_distributed) or (is_distributed and rank == 0)
                )
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                if is_distributed:
                    # if distributed training the model is an instance of the DistributedDataParallel class
                    # and we need to unpack it from it.
                    vision_module = self.trainer.game.module.vision_module
                else:
                    vision_module = self.trainer.game.vision_module
                if self.shared:
                    torch.save(
                        vision_module.encoder,
                        self.trainer.checkpoint_path / "vision_module_shared.pt"
                    )
                else:
                    torch.save(
                        vision_module.encoder,
                        self.trainer.checkpoint_path / "vision_module_sender.pt"
                    )
                    torch.save(
                        vision_module.encoder_recv,
                        self.trainer.checkpoint_path / "vision_module_recv.pt"
                    )


class DistributedSamplerEpochSetter(Callback):
    """A callback that sets the right epoch of a DistributedSampler instance."""
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch):
        # just being cautious here given that non distributed jobs won't have probaly have distributed sampler set
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)

    def on_test_begin(self, epoch):
        # just being cautious here given that non distributed jobs won't have probaly have distributed sampler set
        if self.trainer.distributed_context.is_distributed and self.trainer.validation_data:
            self.trainer.validation_data.sampler.set_epoch(epoch)
