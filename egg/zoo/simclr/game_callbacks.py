# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self, is_distributed: bool = False, rank: int = 0):
        super().__init__()

        self.is_distributed = is_distributed
        self.rank = rank

        self.best_acc = -1.
        self.best_loss = float("inf")
        self.best_epoch = -1

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_acc:
            self.best_acc = logs.aux["acc"].mean().item()
            self.best_epoch = epoch
            self.best_loss = _loss

    def on_train_end(self):
        if (not self.is_distributed) or (self.is_distributed and self.rank == 0):
            print(f"BEST: epoch {self.best_epoch}, acc: {self.best_acc}, loss: {self.best_loss}")


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""
    def __init__(
        self,
        shared: bool,
        is_distributed: bool = False,
        rank: int = 0
    ):
        super().__init__()

        self.shared = shared
        self.is_distributed = is_distributed
        self.rank = rank

    def on_train_end(self):
        if hasattr(self.trainer, "checkpoint_path"):
            if (not self.is_distributed) or (self.is_distributed and self.rank == 0):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                if self.is_distributed:
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
        self.trainer.train_data.sampler.set_epoch(epoch)
