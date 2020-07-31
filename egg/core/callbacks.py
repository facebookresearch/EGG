# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
import pathlib
from typing import Dict, Any, Union,  NamedTuple, List

try:
    import editdistance  # package to install https://pypi.org/project/editdistance/0.3.1/
except ImportError:
    print('Please install editdistance package: `pip install editdistance`. '
          'It is used for calculating topographic similarity.')
from scipy.spatial import distance
from scipy.stats import spearmanr
import torch

from egg.core.util import get_summary_writer
from .interaction import Interaction


class Callback:

    def on_train_begin(self, trainer_instance: 'Trainer'):
        self.trainer = trainer_instance
        self.epoch_counter = self.trainer.start_epoch

    def on_train_end(self):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self, loss: float, logs: Interaction):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, loss: float, logs: Interaction):
        pass


class ConsoleLogger(Callback):

    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: Interaction, mode: str):
        dump = dict(loss=loss) 
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=self.epoch_counter))
            output_message = json.dumps(dump)
        else:
            output_message = ', '.join(sorted([f'{k}={v}' for k, v in dump.items()]))
            output_message = f'{mode}: epoch {self.epoch_counter}, loss {loss}, ' + output_message
        print(output_message, flush=True)

    def on_test_end(self, loss: float, logs: Interaction):
        self.aggregate_print(loss, logs, 'test')

    def on_epoch_end(self, loss: float, logs: Interaction):
        self.epoch_counter += 1

        if not self.print_train_loss: return
        self.aggregate_print(loss, logs, 'train')


class TensorboardLogger(Callback):

    def __init__(self, writer=None):
        if writer:
            self.writer = writer
        else:
            self.writer = get_summary_writer()
        self.epoch_counter = 0

    def on_test_end(self, loss: float, logs: Interaction):
        self.writer.add_scalar(tag=f'test/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.aux.items():
            self.writer.add_scalar(tag=f'test/{k}', scalar_value=v.mean(), global_step=self.epoch_counter)

    def on_epoch_end(self, loss: float, logs: Interaction):
        self.writer.add_scalar(tag=f'train/loss', scalar_value=loss, global_step=self.epoch_counter)
        for k, v in logs.aux.items():
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=v.mean(), global_step=self.epoch_counter)
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
        return Checkpoint(epoch=self.epoch_counter,
                          model_state_dict=self.trainer.game.state_dict(),
                          optimizer_state_dict=self.trainer.optimizer.state_dict())


class TopographicSimilarity(Callback):
    distances = {'edit': lambda x, y: editdistance.eval(x, y) / (len(x) + len(y)) / 2,
                 'cosine': distance.cosine,
                 'hamming':distance.hamming,
                 'jaccard': distance.jaccard,
                 'euclidean': distance.euclidean,
                 }

    def __init__(self,
                 sender_input_distance_fn='cosine',
                 message_distance_fn='edit',
                 compute_topsim_train_set=False,
                 compute_topsim_test_set=True):

        self.sender_input_distance_fn = self.distances.get(sender_input_distance_fn, None)
        self.message_distance_fn = self.distances.get(message_distance_fn, None)
        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set

        assert self.sender_input_distance_fn and self.message_distance_fn, f"Cannot recognize {sender_input_distance_fn} or {message_distance_fn} distances"
        assert compute_topsim_train_set or compute_topsim_test_set

    def on_test_end(self, loss: float, logs: Interaction):
        if self.compute_topsim_test_set:
            self.compute_similarity(sender_input=logs.sender_input, messages=logs.message)

    def on_epoch_end(self, loss: float, logs: Interaction):
        if self.compute_topsim_train_set:
            self.compute_similarity(sender_input=logs.sender_input, messages=logs.message)

    def compute_similarity(self, sender_input, messages):
        def compute_distance(_list, distance):
            return [distance(el1, el2)
                        for i, el1 in enumerate(_list[:-1])
                        for j, el2 in enumerate(_list[i+1:])
                    ]

        messages = [msg.tolist() for msg in messages]

        input_dist = compute_distance(sender_input.numpy(), self.sender_input_distance_fn)
        message_dist = compute_distance(messages, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist, nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim))
        print(output_message, flush=True)
