# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pathlib
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

import torch
import wandb
from rich.columns import Columns
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from egg.core.interaction import Interaction
from egg.core.util import get_summary_writer


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

    def on_validation_begin(self, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        pass

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
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

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
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

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
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


class WandbLogger(Callback):
    def __init__(
        self,
        opts: Union[argparse.ArgumentParser, Dict, str, None] = None,
        project: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        # This callback logs to wandb the interaction as they are stored in the leader process.
        # When interactions are not aggregated in a multigpu run, each process will store
        # its own Interaction object in logs. For now, we leave to the user handling this case by
        # subclassing WandbLogger and implementing a custom logic since we do not know a priori
        # what type of data are to be logged.
        self.opts = opts

        wandb.init(project=project, id=run_id, **kwargs)
        wandb.config.update(opts)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance
        wandb.watch(self.trainer.game, log="all")

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"batch_loss": loss}, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"train_loss": loss, "epoch": epoch}, commit=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb({"validation_loss": loss, "epoch": epoch}, commit=True)


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
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]


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
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            filename = f"{self.prefix}_{epoch}" if self.prefix else str(epoch)
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
        if len(self.get_checkpoint_files()) > self.max_checkpoints:
            self.remove_oldest_checkpoint()
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )
        if self.trainer.distributed_context.is_distributed:
            game = self.trainer.game.module
        else:
            game = self.trainer.game
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
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
        checkpoints = self.natural_sort(self.get_checkpoint_files())
        os.remove(os.path.join(self.checkpoint_path, checkpoints[0]))


class InteractionSaver(Callback):
    def __init__(
        self,
        train_epochs: Optional[List[int]] = None,
        test_epochs: Optional[List[int]] = None,
        checkpoint_dir: str = "",
        aggregated_interaction: bool = True,
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

        if checkpoint_dir:
            self.checkpoint_dir = pathlib.Path(checkpoint_dir) / "interactions"
        else:
            self.checkpoint_dir = pathlib.Path("./interactions")

        self.aggregated_interaction = aggregated_interaction

    @staticmethod
    def dump_interactions(
        logs: Interaction,
        mode: str,
        epoch: int,
        rank: int,
        dump_dir: str = "./interactions",
    ):
        dump_dir = pathlib.Path(dump_dir) / mode / f"epoch_{epoch}"
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(logs, dump_dir / f"interaction_gpu{rank}")

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.test_epochs:
            if (
                not self.aggregated_interaction
                or self.trainer.distributed_context.is_leader
            ):
                rank = self.trainer.distributed_context.rank
                self.dump_interactions(
                    logs, "validation", epoch, rank, self.checkpoint_dir
                )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.test_epochs:
            if (
                not self.aggregated_interaction
                or self.trainer.distributed_context.is_leader
            ):
                rank = self.trainer.distributed_context.rank
                self.dump_interactions(logs, "train", epoch, rank, self.checkpoint_dir)


class CustomProgress(Progress):
    class CompletedColumn(ProgressColumn):
        def render(self, task):
            """Calculate common unit for completed and total."""
            download_status = f"{int(task.completed)}/{int(task.total)} btc"
            return Text(download_status, style="progress.download")

    class TransferSpeedColumn(ProgressColumn):
        """Renders human readable transfer speed."""

        def render(self, task):
            """Show data transfer speed."""
            speed = task.speed
            if speed is None:
                return Text("?", style="progress.data.speed")
            speed = f"{speed:,.{2}f}"
            return Text(f"{speed} btc/s", style="progress.data.speed")

    def __init__(self, *args, use_info_table: bool = True, **kwargs):
        super(CustomProgress, self).__init__(*args, **kwargs)

        self.info_table = Table(show_footer=False)
        self.info_table.add_column("Phase")

        self.test_style = "black on white"
        self.train_style = "white on black"
        self.use_info_table = use_info_table

    def add_info_table_cols(self, new_cols):
        """
        Add cols from ordered dict if not present in info_table
        """

        cols = set([x.header for x in self.info_table.columns])
        missing = set(new_cols) - cols
        if len(missing) == 0:
            return

        # iterate on new_cols since they are in order
        for c in new_cols:
            if c in missing:
                self.info_table.add_column(c)

    def update_info_table(self, aux: Dict[str, float], phase: str):
        """
        Update the info_table with the latest results
        :param aux:
        :param phase: either 'train' or 'test'
        """

        self.add_info_table_cols(aux.keys())
        epoch = aux.pop("epoch")
        aux = OrderedDict((k, f"{v:4.3f}") for k, v in aux.items())
        if phase == "train":
            st = self.train_style
        else:
            st = self.test_style
        self.info_table.add_row(phase, str(epoch), *list(aux.values()), style=st)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Display progress together with info table"""

        # this method is called once before the init, so check if the attribute is present
        if hasattr(self, "use_info_table"):
            use_table = self.use_info_table
            info_table = self.info_table
        else:
            use_table = False
            info_table = Table()

        if use_table:
            task_table = self.make_tasks_table(self.tasks)
            rendable = Columns((info_table, task_table), align="left", expand=True)
        else:
            rendable = self.make_tasks_table(self.tasks)

        yield rendable


class ProgressBarLogger(Callback):
    """
    Displays a progress bar with information about the current epoch and the epoch progression.
    """

    def __init__(
        self,
        n_epochs: int,
        train_data_len: int = 0,
        test_data_len: int = 0,
        use_info_table: bool = True,
    ):
        """
        :param n_epochs: total number of epochs
        :param train_data_len: length of the dataset generation for training
        :param test_data_len: length of the dataset generation for testing
        :param use_info_table: true to add an information table on top of the progress bar
        """

        self.n_epochs = n_epochs
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.progress = CustomProgress(
            TextColumn(
                "[bold]Epoch {task.fields[cur_epoch]}/{task.fields[n_epochs]} | [blue]{task.fields[mode]}",
                justify="right",
            ),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            CustomProgress.CompletedColumn(),
            "•",
            CustomProgress.TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            use_info_table=use_info_table,
        )

        self.progress.start()
        self.train_p = self.progress.add_task(
            description="",
            mode="Train",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.train_data_len,
        )
        self.test_p = self.progress.add_task(
            description="",
            mode="Test",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.test_data_len,
        )

    @staticmethod
    def build_od(logs, loss, epoch):
        od = OrderedDict()
        od["epoch"] = epoch
        od["loss"] = loss
        aux = {k: float(torch.mean(v)) for k, v in logs.aux.items()}
        od.update(aux)

        return od

    def on_epoch_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )
        self.progress.start_task(self.train_p)
        self.progress.update(self.train_p, visible=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.stop_task(self.train_p)
        self.progress.update(self.train_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.train_data_len == 0:
            self.train_data_len = self.progress.tasks[self.train_p].completed

        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )

        od = self.build_od(logs, loss, epoch)
        self.progress.update_info_table(od, "train")

    def on_validation_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        self.progress.start_task(self.test_p)
        self.progress.update(self.test_p, visible=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.stop_task(self.test_p)
        self.progress.update(self.test_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.test_data_len == 0:
            self.test_data_len = self.progress.tasks[self.test_p].completed

        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        od = self.build_od(logs, loss, epoch)
        self.progress.update_info_table(od, "test")

    def on_train_end(self):
        self.progress.stop()

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training:
            self.progress.update(self.train_p, refresh=True, advance=1)
        else:
            self.progress.update(self.test_p, refresh=True, advance=1)
