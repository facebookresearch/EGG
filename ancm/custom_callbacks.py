# import logging
# from tqdm import tqdm
import torch
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

from rich.live import Live
from rich.columns import Columns
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from egg.core import Callback
from egg.core.interaction import Interaction


class EpochProgress(Progress):
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
            speed = f"{1/speed:,.{2}f}"
            return Text(f"{speed} s/ep", style="progress.data.speed")

    def __init__(self, *args, **kwargs):
        super(EpochProgress, self).__init__(*args, **kwargs)


class CustomProgressBarLogger(Callback):
    """
    Displays a progress bar with information about the current epoch and the epoch progression.
    """

    def __init__(
        self,
        n_epochs: int,
        train_data_len: int = 0,
        test_data_len: int = 0,
        print_train_metrics = True,
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
        self.print_train_metrics = print_train_metrics

        self.progress = EpochProgress(
            TextColumn(
                "[bold]{task.fields[cur_epoch]}/{task.fields[n_epochs]}",
                justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            EpochProgress.TransferSpeedColumn(), "•",
            TimeElapsedColumn(), "•",
            TimeRemainingColumn(elapsed_when_finished=True))
        self.live = Live(self.generate_live_table())
        self.console = self.live.console

        self.live.start()

        self.p = self.progress.add_task(
            description="[cyan]Training",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=True,
            visible=True,
            total=self.n_epochs)
        self.history = dict()

        self.eval_style = "white on black"
        self.train_style = "grey37 on black"

    def get_row(self, od, header=False):
        row = Table(expand=True, box=None, show_header=header, show_footer=False)
        for colname in od.keys():
            row.add_column(
                colname,
                justify='left' if colname in ('phase', 'epoch')  else 'right',
                ratio=1)
        if not header:
            row.add_row(
                str(od.pop('epoch')),
                *list(v if isinstance(v, str) else f"{v: 4.4f}" for v in od.values()),
                style=self.train_style if od['phase'] == 'train' else self.eval_style)
        return row

    @staticmethod
    def build_od(logs, loss, epoch, phase):
        od = OrderedDict()
        od["epoch"] = epoch
        od['phase'] = phase
        od["loss"] = loss
        aux = {k: float(torch.mean(v)) for k, v in logs.aux.items()}
        od.update(aux)
        return od

    def generate_live_table(self, od=None):
        live_table = Table.grid(expand=True)
        if od:
            header = self.get_row(od=od, header=True)
            live_table.add_row(header)
        live_table.add_row(self.progress)
        return live_table

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        od = self.build_od(logs, loss, epoch, 'train')
        if self.print_train_metrics:
            if epoch == 1:
                self.live.update(self.generate_live_table(od))
            row = self.get_row(od)
            self.console.print(row)

    def on_validation_begin(self, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.update(self.p, refresh=True, advance=1, cur_epoch=epoch)

        # if the datalen is zero update with the one epoch just ended
        if self.test_data_len == 0:
            self.test_data_len = self.progress.tasks[self.p].completed

        od = self.build_od(logs, loss, epoch, 'eval ')
        if not self.print_train_metrics and epoch == 1:
            self.live.update(self.generate_live_table(od))
        row = self.get_row(od)
        self.console.print(row)

    def on_train_end(self):
        self.live.stop()

    def on_batch_end(self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True):
        pass
