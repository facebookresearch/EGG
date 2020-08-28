# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
import torch.distributed as distrib

from .util import get_opts, move_to
from .callbacks import Callback, ConsoleLogger, Checkpoint, CheckpointSaver, TensorboardLogger
from .interaction import Interaction
from .distributed import maybe_init_distributed, get_preemptive_checkpoint_dir

class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None,
            grad_norm: float = None,
            aggregate_interaction_logs: bool = True
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print('# Distributed context: ', self.distributed_context)

        if self.distributed_context.is_leader and not any(isinstance(x, CheckpointSaver) for x in self.callbacks):
            if common_opts.preemptable:
                assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                    else pathlib.Path(common_opts.checkpoint_dir)

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(checkpoint_path=self.checkpoint_path, checkpoint_freq=common_opts.checkpoint_freq)
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert common_opts.tensorboard_dir, 'tensorboard directory has to be specified'
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        if self.distributed_context.is_distributed:
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            self.game.to(device_id)
            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper. This wrapper would
            #    sync gradients of the underlying tensors - which are the ones that optimizer holds itself. 
            #    As a result it seems to work, but only because DDP doesn't take any tensor ownership.
            #    

            self.game = torch.nn.parallel.DistributedDataParallel(self.game,
                                                  device_ids=[device_id],
                                                  output_device=device_id)


        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)

    def eval(self):
        mean_loss = 0.0
        interactions = []

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss, interaction = self.game(*batch)
                if self.distributed_context.is_distributed and self.aggregate_interaction_logs:
                    interaction = Interaction.gather_distributed_interactions(interaction)
                interaction = interaction.to('cpu')
                mean_loss += optimized_loss
                n_batches += 1
                interactions.append(interaction)
        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        for batch in self.train_data:
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            optimized_loss, interaction = self.game(*batch)
            optimized_loss.backward()

            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.game.parameters(), self.grad_norm)

            self.optimizer.step()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            if self.distributed_context.is_distributed and self.aggregate_interaction_logs:
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to('cpu')
            interactions.append(interaction)

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch+1)

            train_loss, train_interaction = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_interaction, epoch+1)

            if self.validation_data is not None and self.validation_freq > 0 and (epoch+1) % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin(epoch+1)
                validation_loss, validation_interaction = self.eval()

                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, validation_interaction, epoch+1)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.start_epoch = checkpoint.epoch

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
