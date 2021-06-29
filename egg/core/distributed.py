# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
import subprocess
import uuid
from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True, repr=True, eq=True, unsafe_hash=True)
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    mode: str

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def maybe_init_distributed(args) -> DistributedContext:
    assert not hasattr(
        args, "distributed_context"
    ), "distributed context is already initialized?!"
    # default, non-distributed context
    context = DistributedContext(
        is_distributed=False, rank=0, local_rank=0, world_size=1, mode="none"
    )

    launch_keys = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]
    slurm_keys = [
        "SLURM_LOCALID",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_NODEID",
        "SLURM_JOB_NODELIST",
    ]

    # is it torch.distributed.launch?
    if all(key in os.environ for key in launch_keys):
        init_method = "env://"
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        context = DistributedContext(
            is_distributed=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            mode="launch",
        )
        dist.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=rank
        )
    # is it slurm?
    elif all(key in os.environ for key in slurm_keys):
        init_method = "env://"
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        leader_addr = hostnames.split()[0].decode("utf-8")

        os.environ["MASTER_ADDR"] = leader_addr
        os.environ["MASTER_PORT"] = str(args.distributed_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        if world_size > 1:
            # no point in being distributed if it is alone
            context = DistributedContext(
                is_distributed=True,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                mode="slurm",
            )
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

    return context


# should be part of the dustributed context
def get_preemptive_checkpoint_dir(checkpoint_root):
    if "SLURM_JOB_ID" not in os.environ:
        print("Preemption flag set, but I am not running under SLURM?")

    # start assuming the job runs as an array
    # fallback, if not
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if job_id is None or task_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
        task_id = os.environ.get("SLURM_PROCID", 0)

    d = pathlib.Path(checkpoint_root) / f"{job_id}_{task_id}"
    d.mkdir(exist_ok=True)

    return d
