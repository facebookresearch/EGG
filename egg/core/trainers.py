# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
from .util import get_opts, move_to, get_summary_writer
import pathlib
import os
import uuid


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _div_dict(d, n):
    result = dict(d)
    for k in result:
        result[k] /= n
    return result


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(self, game, optimizer, train_data, validation_data=None, device=None, epoch_callback=None,
                 as_json=False, early_stopping=None, print_train_loss=False):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param epoch_callback: A callable that would be called at the end of each epoch (after validation, can be None).
        :param as_json: Output validation statistics as json
        :param early_stopping: An instance that defines the stopping logic. Called after every run on validation data;
            see early_stopping.py for an example.
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.epoch_callback = epoch_callback
        self.checkpoint_freq = common_opts.checkpoint_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        self.as_json = as_json
        self.early_stopping = early_stopping
        self.print_train_loss = print_train_loss

        self.epoch = 0

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss, rest = self.game(*batch)
                mean_loss += optimized_loss
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss, mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            optimized_loss, rest = self.game(*batch)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            self.optimizer.step()

            n_batches += 1
            mean_loss += optimized_loss

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss, mean_rest

    def train(self, n_epochs):
        def _message(mode, epoch, loss, rest, as_json):
            writer = get_summary_writer()
            if writer:
                writer.add_scalar(tag=f'{mode}/loss', scalar_value=loss.mean(), global_step=epoch)
                for k, v in rest.items():
                    writer.add_scalar(tag=f'{mode}/{k}', scalar_value=v, global_step=epoch)
            if as_json:
                dump = dict(mode=mode, epoch=epoch, loss=loss.mean().item())
                for k, v in rest.items(): dump[k] = v.item() if hasattr(v, 'item') else v
                output_message = json.dumps(dump)
            else:
                output_message = f'{mode}: epoch {self.epoch}, loss {loss},  {rest}'
            print(output_message, flush=True)

        while self.epoch < n_epochs:
            train_loss, train_rest = self.train_epoch()

            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(self)

            if self.checkpoint_freq > 0 and (self.epoch % self.checkpoint_freq == 0) and self.checkpoint_path:
                self.save_checkpoint()

            if self.print_train_loss:
                _message('train', self.epoch, train_loss, train_rest, self.as_json)

            if self.validation_data is not None and self.validation_freq > 0 and self.epoch % self.validation_freq == 0:
                validation_loss, rest = self.eval()
                _message('test', self.epoch, validation_loss, rest, self.as_json)

                if self.early_stopping:
                    self.early_stopping.update_values(validation_loss, rest, train_loss, rest, self.epoch)
                    if self.early_stopping.should_stop(): break

        if self.checkpoint_path:
            self.save_checkpoint()

    def save_checkpoint(self):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True)
        path = self.checkpoint_path / f'{self.epoch}.tar'
        torch.save(self.get_state(), path)

    def get_state(self):
        return dict(epoch=self.epoch,
                model_state_dict=self.game.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict())

    def load(self, checkpoint):
        self.game.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)



