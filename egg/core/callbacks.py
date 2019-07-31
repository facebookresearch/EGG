# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, Any, Union,  NamedTuple
import pathlib

import torch

from egg.core.util import get_summary_writer


class Callback:
    trainer: 'Trainer'

    def on_train_begin(self, trainer_instance: 'Trainer'):
        self.trainer = trainer_instance

    def on_train_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        pass


class ConsoleLogger(Callback):

    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.as_json:
            dump = dict(mode='test', epoch=self.epoch_counter, loss=self._get_metric(loss))
            for k, v in logs.items():
                dump[k] = self._get_metric(v)
            output_message = json.dumps(dump)
        else:
            output_message = f'test: epoch {self.epoch_counter}, loss {loss},  {logs}'
        print(output_message, flush=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.print_train_loss:
            if self.as_json:
                dump = dict(mode='train', epoch=self.epoch_counter, loss=self._get_metric(loss))
                for k, v in logs.items():
                    dump[k] = self._get_metric(v)
                output_message = json.dumps(dump)
            else:
                output_message = f'train: epoch {self.epoch_counter}, loss {loss},  {logs}'
            print(output_message, flush=True)
        self.epoch_counter += 1

    def _get_metric(self, metric: Union[torch.Tensor, float]) -> float:
        if torch.is_tensor(metric) and metric.dim() > 1:
            return metric.mean().item()
        elif torch.is_tensor(metric):
            return metric.item()
        elif type(metric) == float:
            return metric
        else:
            raise TypeError('Metric must be either float or torch.Tensor')


class TensorboardLogger(Callback):

    def __init__(self, writer=None):
        if writer:
            self.writer = writer
        else:
            self.writer = get_summary_writer
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'test/loss', scalar_value=loss.mean(), global_step=self.epoch)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'test/{k}', scalar_value=v, global_step=self.epoch)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'train/loss', scalar_value=loss.mean(), global_step=self.epoch)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=v, global_step=self.epoch)
        self.epoch_counter += 1

    def on_train_end(self):
        self.writer.close()


class TemperatureUpdater(Callback):

    def __init__(self, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.agent = agent
        assert hasattr(agent, 'temperature'), 'Agent must have a `temperature` attribute'
        assert not isinstance(agent.temperature, torch.nn.Parameter), \
            'When using TemperatureUpdater, `temperature` cannot be trainable'
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.epoch_counter % self.update_frequency == 0:
            self.agent.temperature = max(self.minimum, self.agent.temperature * self.decay)
        self.epoch_counter += 1


class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]


class CheckpointSaver(Callback):

    def __init__(
            self,
            checkpoint_path: Union[str, pathlib.Path],
            checkpoint_freq: int = 1,
            prefix: str = ''
    ):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.checkpoint_freq > 0 and (self.epoch_counter % self.checkpoint_freq == 0):
            filename = f'{self.prefix}_{self.epoch_counter}' if self.prefix else str(self.epoch_counter)
            self.save_checkpoint(filename=filename)
        self.epoch_counter += 1

    def on_train_end(self):
        self.save_checkpoint(filename=f'{self.prefix}_final' if self.prefix else 'final')

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True)
        path = self.checkpoint_path / f'{filename}.tar'
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        return Checkpoint(epoch=self.epoch_counter,
                          model_state_dict=self.trainer.game.state_dict(),
                          optimizer_state_dict=self.trainer.optimizer.state_dict())
