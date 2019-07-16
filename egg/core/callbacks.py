import json
from typing import Dict, Any

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
            dump = dict(mode='test', epoch=self.epoch_counter, loss=loss.mean().item())
            for k, v in logs.items():
                dump[k] = v.item() if hasattr(v, 'item') else v
            output_message = json.dumps(dump)
        else:
            output_message = f'test: epoch {self.epoch_counter}, loss {loss},  {logs}'
        print(output_message, flush=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        self.epoch_counter += 1
        if self.print_train_loss:
            if self.as_json:
                dump = dict(mode='train', epoch=self.epoch_counter, loss=loss.mean().item())
                for k, v in logs.items():
                    dump[k] = v.item() if hasattr(v, 'item') else v
                output_message = json.dumps(dump)
            else:
                output_message = f'train: epoch {self.epoch_counter}, loss {loss},  {logs}'
            print(output_message, flush=True)


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
        self.epoch_counter += 1
        self.writer.add_scalar(tag=f'train/loss', scalar_value=loss.mean(), global_step=self.epoch)
        for k, v in logs.items():
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=v, global_step=self.epoch)

    def on_train_end(self):
        self.writer.close()
