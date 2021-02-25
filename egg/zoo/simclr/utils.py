# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import torch

from egg.core import Callback, Interaction


class BestStatsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_acc = -1.
        self.best_loss = float("inf")
        self.best_epoch = -1

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if logs.aux["acc"].mean().item() > self.best_acc:
            self.best_acc = logs.aux["acc"].mean().item()
            self.best_epoch = epoch
            self.best_loss = _loss

    def on_train_end(self):
        print(f"BEST: epoch {self.best_epoch}, acc: {self.best_acc}, loss: {self.best_loss}")


class VisionModelSaver(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = pathlib.Path(path)

    def on_train_end(self):
        if self.self.trainer.game.vision_module.shared:
            torch.save(self.trainer.game.vision_module.encoder, self.path / "vision_module_shared.pt")
        if self.self.trainer.game.vision_module.shared:
            torch.save(self.trainer.game.vision_module.encoder, self.path / "vision_module_sender.pt")
            torch.save(self.trainer.game.vision_module.encoder_recv, self.path / "vision_module_recv.pt")
