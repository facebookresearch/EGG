# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib
import sys
import torch


class SlurmWrapper:
    """
    We assume that checkpointing is done within trainer, each epoch.
    """
    def __init__(self, runnable):
        self.runnable = runnable
        self.args = None

    def __call__(self, args):
        self.args = args
        print(f'# launching {json.dumps(args)}', flush=True)
        self.runnable(args)

    def checkpoint(self, _something):
        import submitit

        training_callable = SlurmWrapper(self.runnable)
        return submitit.helpers.DelayedSubmission(training_callable, self.args)


class ConcurrentWrapper:
    def __init__(self, runnable, log_dir, job_id):
        self.runnable = runnable
        self.args = None
        self.log_dir = log_dir
        self.job_id = job_id

    def __call__(self, args):
        stdout_path = pathlib.Path(self.log_dir) / f'{self.job_id}.out'
        self.stdout = open(stdout_path, 'w')

        stderr_path = pathlib.Path(self.log_dir) / f'{self.job_id}.err'
        self.stderr = open(stderr_path, 'w')

        sys.stdout = self.stdout
        sys.stderr = self.stderr
        cuda_id = -1
        n_devices = torch.cuda.device_count()
        if n_devices > 0:
            cuda_id = self.job_id % n_devices
        print(f'# {json.dumps(args)}', flush=True)

        with torch.cuda.device(cuda_id):
            self.runnable(args)
