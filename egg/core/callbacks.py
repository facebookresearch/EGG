# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, NamedTuple, Union

import torch

from egg.core.util import get_summary_writer

from .interaction import Interaction


class Callback:
    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance

    def on_train_end(self):
        pass

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        pass

    def on_test_begin(self, epoch: int):
        pass

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        pass

    def on_batch_end(self, logs: Interaction, loss: float, batch_id: int):
        pass


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

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)


class TensorboardLogger(Callback):
    def __init__(self, writer=None):
        if writer:
            self.writer = writer
        else:
            self.writer = get_summary_writer()

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.writer.add_scalar(tag="test/loss", scalar_value=loss, global_step=epoch)
        for k, v in logs.aux.items():
            self.writer.add_scalar(
                tag=f"test/{k}", scalar_value=v.mean(), global_step=epoch
            )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=epoch)
        for k, v in logs.aux.items():
            self.writer.add_scalar(
                tag=f"train/{k}", scalar_value=v.mean(), global_step=epoch
            )

    def on_train_end(self):
        self.writer.close()


class TemperatureUpdater(Callback):
    def __init__(self, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.agent = agent
        assert hasattr(
            agent, "temperature"
        ), "Agent must have a `temperature` attribute"
        assert not isinstance(
            agent.temperature, torch.nn.Parameter
        ), "When using TemperatureUpdater, `temperature` cannot be trainable"
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch % self.update_frequency == 0:
            self.agent.temperature = max(
                self.minimum, self.agent.temperature * self.decay
            )


class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]


class CheckpointSaver(Callback):
    def __init__(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        checkpoint_freq: int = 1,
        prefix: str = "",
        max_checkpoints: int = sys.maxsize,
    ):
        """Saves a checkpoint file for training.
            :param checkpoint_path:  path to checkpoint directory, will be created if not present
            :param checkpoint_freq:  Number of epochs for checkpoint saving
            :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
            :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.max_checkpoints = max_checkpoints
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.epoch_counter = epoch
        if self.checkpoint_freq > 0 and (
            self.epoch_counter % self.checkpoint_freq == 0
        ):
            filename = (
                f"{self.prefix}_{self.epoch_counter}"
                if self.prefix
                else str(self.epoch_counter)
            )
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        if len(self.get_checkpoint_files()) > self.max_checkpoints_keep:
            self.remove_old_chk()
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=self.trainer.game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
        )

    def get_checkpoint_files(self):
        """
        Return a list of the files in the checkpoint dir
        """
        return [name for name in os.listdir(self.checkpoint_path) if ".tar" in name]

    @staticmethod
    def natural_sort(to_sort):
        """
        Sort a list of files naturally
        E.g. [file1,file4,file32,file2] -> [file1,file2,file4,file32]
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(to_sort, key=alphanum_key)

    def remove_oldest_checkpoint(self):
        """
        Remove the oldest checkpoint from the dir
        """
        chk_points = self.natural_sort(self.get_checkpoint_files())
        to_remove = chk_points[0]
        os.remove(os.path.join(self.checkpoint_path, to_remove))


class InteractionSaver(Callback):
    def __init__(
        self,
        train_epochs: List = None,
        test_epochs: List = None,
        folder_path: str = "./interactions",
    ):
        if isinstance(train_epochs, list):
            assert all(map(lambda x: x > 0, train_epochs))
            self.train_epochs = train_epochs
        else:
            self.train_epochs = []
        if isinstance(test_epochs, list):
            assert all(map(lambda x: x > 0, test_epochs))
            self.test_epochs = test_epochs
        else:
            self.test_epochs = []

        self.folder_path = (
            pathlib.Path(folder_path) / time.strftime("%Y_%m_%d_%H_%M_%S")
        ).expanduser()

    @staticmethod
    def dump_interactions(
        logs: Interaction, mode: str, epoch: int, dump_dir: str = "./interactions"
    ):
        dump_dir = pathlib.Path(dump_dir) / mode
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(logs, dump_dir / f"interactions_epoch{epoch}")

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.test_epochs:
            self.dump_interactions(logs, "validation", epoch, self.folder_path)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.train_epochs:
            self.dump_interactions(logs, "train", epoch, self.folder_path)
