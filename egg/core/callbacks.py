# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import List, Dict, Any, Union, NamedTuple, Tuple, Callable, Iterable
import pathlib

import torch

from egg.core.util import get_summary_writer


Games = List[torch.nn.Module]


class Callback:

    def on_train_begin(self, trainer_instance: 'Trainer'):
        self.trainer = trainer_instance
        self.epoch_counter = self.trainer.start_epoch

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


class Evaluator(Callback):

    def __init__(self,
                 modules: List[Union[Games, torch.nn.Module],
                 dataset_gn: Union[Iterable[Any], Generator[Any, None, None]],
                 metric_fns: List[Tuple[str, Callable[torch.nn.Module, Any]]]),
                 device: torch.device,
                 output_file_name: str = 'metrics.json',
                 output_file_path: Union[str, pathlib.Path] = None):

        self.modules = modules
        self.dataset_gn = dataset_gn
        self.metric_fns = metric_fns
        self.device = device

        self.stats: List[Dict[str, Any]] = defaultdict(float)
        assert len(modules) == len(metric_fns)

        self.output_file_path = output_file_path
        if output_file_path:
            self.output_file_path = pathlib.Path(outputfile_path) / output_file_name

    def _div_dict(self, d, n):
        result = dict(d)
        for k in result:
            if isinstance(result[k], dict):
                _div_dict(result[k], n)
            else:
                result[k] /= n
        return result

    def _add_dict(self, a, b):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + v
        return result

    def _add_metric(self, a, b, metric_name):
        result = dict(a)
        if isinstance(b, dict):
            self._add_dict(result, b)
        else:
            result[metric_name] = result.get(metric_name, 0) + b
        return result

    def evaluate(self):
        for module in self.modules:
            module.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch = move_to(batch, self.device)
                for module, (metric_name, metric_fn) in zip(modules, metric_fns):
                    if isinstance(module, list):
                        res_module_list = defaultdict(float)
                        for j, single_module in enumerate(module):
                            # passing j as module idx as it might be used by the metric function
                            self._add_metric(res_module_list[j], metric_fn(module_output, batch, module_idx=j), metric_name=j)
                        res = self._div_dict(res_module_list, j+1) if isinstance(res_module_list, dict) else res / (j+1)

                        self._add_metric(self.stats, res, metric_name)
                    else:
                        self._add_metric(self.stats, metric_fn(module_output), metric_name)

        for module in self.modules:
            module.train()

        self._div_dict(self.stats, i+1)

    def dump(self):
        with open(self.output_file_path, 'w') as fd:
            for stat in self.stats:
                json.dump(stat, fd)
                w.write('\n')

    def on_test_end(self):
        self.evaluate()
        if self.output_file_path:
            self.dump()


class ConsoleLogger(Callback):

    def __init__(self, print_train_loss=False, as_json=False, output_file=None):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.output_file = output_file

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        if self.as_json:
            dump = dict(mode='test', epoch=self.epoch_counter, loss=self._get_metric(loss))
            for k, v in logs.items():
                dump[k] = self._get_metric(v)
            output_message = json.dumps(dump)
        else:
            output_message = f'test: epoch {self.epoch_counter}, loss {loss},  {logs}'

        if self.output_file:
            with open(self.output_file, 'a') as fd:
                fd.write('{output_message}\n')
        print(output_message, flush=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.epoch_counter += 1

        if self.print_train_loss:
            if self.as_json:
                dump = dict(mode='train', epoch=self.epoch_counter, loss=self._get_metric(loss))
                for k, v in logs.items():
                    dump[k] = self._get_metric(v)
                output_message = json.dumps(dump)
            else:
                output_message = f'train: epoch {self.epoch_counter}, loss {loss},  {logs}'
            if self.output_file:
                with open(self.output_file, 'a') as fd:
                    fd.write(f'{output_message}\n')
            print(output_message, flush=True)

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
            self.writer = get_summary_writer()
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'test/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'test/{k}', scalar_value=v, global_step=self.epoch_counter)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.writer.add_scalar(tag=f'train/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=v, global_step=self.epoch_counter)
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
    model_state_dicts: List[Dict[str, Any]]
    optimizer_state_dicts: List[Dict[str, Any]]


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

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.epoch_counter += 1
        if self.checkpoint_freq > 0 and (self.epoch_counter % self.checkpoint_freq == 0):
            filename = f'{self.prefix}_{self.epoch_counter}' if self.prefix else str(self.epoch_counter)
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(filename=f'{self.prefix}_final' if self.prefix else 'final')

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        path = self.checkpoint_path / f'{filename}.tar'
        torch.save(self.get_checkpoint(), path)


    def get_checkpoint(self):
        model_state_dicts = dict( [i, game.state_dict()] for i, game in enumerate(self.trainer.games))
        optimizers_state_dicts = dict( [i, optimizer.state_dict()] for i, optimizer in enumerate(self.trainer.optimizers))
        return Checkpoint(epoch=self.epoch_counter,
                          model_state_dicts=model_state_dicts,
                          optimizer_state_dicts=optimizers_state_dicts)
