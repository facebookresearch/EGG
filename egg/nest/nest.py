# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import pathlib
import time

from egg.nest.common import sweep

if __name__ == "__main__":
    import submitit

    from egg.nest.wrappers import SlurmWrapper

    parser = argparse.ArgumentParser(
        description="nest: a stool-like slurm tool for EGG"
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Game's full classpath to run, e.g. egg.zoo.mnist_autoenc.train",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Json file with sweep params in the stool format."
        "It is possible to specify several files: --sweep file1.json --sweep file2.json",
    )
    parser.add_argument(
        "--py_sweep",
        action="append",
        default=[],
        help="A python module, with a grid() method returning an iterable with grid parameters."
        "It is possible to specify several files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Root folder to save job output",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Don't start jobs, only show the list of commands",
    )
    parser.add_argument("--dry_run", action="store_true", help="Synonym for preview")

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="sbatch name of job. Also used in the output directory",
    )
    parser.add_argument("--tasks", type=int, default=1, help="Number of task per node")
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes required per task"
    )
    parser.add_argument(
        "--partition", type=str, default="devlab", help="Partition requested"
    )
    parser.add_argument("--time", type=int, default=4320, help="Job timeout")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=0,
        help="Checkpoint frequency, imposed on an EGG game, in epochs (used to survive preemption)."
        "Disabled if set to 0.",
    )
    parser.add_argument(
        "--no_preemption",
        action="store_true",
        help="Disable preemption from other processes on SLURM",
    )
    parser.add_argument(
        "--mem_gb", type=int, default=64, help="CPU memory (in GB) required per task"
    )
    parser.add_argument(
        "--constraint",
        type=str,
        help="Constraint (e.g. volta32gb) on the list of jobs to launch",
    )
    parser.add_argument(
        "--comment", type=str, help="Comment on the list of jobs to launch"
    )

    parser.add_argument(
        "--force_requeue",
        action="store_true",
        help="Force job requeue after 1 minute [for debug]",
    )

    parser.add_argument("--array", action="store_true", help="Use SLURM arrays")
    parser.add_argument(
        "--array_parallelism",
        type=int,
        default=128,
        help="Max number of parallel jobs" "executed from the search",
    )

    args = parser.parse_args()

    if not args.game:
        print("--game parameter has to set an EGG-implemented game")
        exit(1)

    if not args.name:
        args.name = args.game.split(".")[-2]

    combinations = []
    for sweep_file in args.sweep:
        combinations.extend(sweep(sweep_file))

    for py_sweep_file in args.py_sweep:
        sweep_module = importlib.import_module(py_sweep_file)
        combinations.extend(sweep_module.grid())

    if args.preview or args.dry_run:
        print(*combinations, sep="\n")
        print(f"Total number of combinations: {len(combinations)}")
        exit()

    if args.checkpoint_dir is None:
        base_dir = pathlib.PosixPath("~/nest")
        args.checkpoint_dir = (
            base_dir / args.name / time.strftime("%Y_%m_%d_%H_%M_%S")
        ).expanduser()
    module = importlib.import_module(args.game)
    executor = submitit.AutoExecutor(folder=args.checkpoint_dir)
    executor.update_parameters(
        timeout_min=args.time,
        slurm_partition=args.partition,
        cpus_per_task=10,
        gpus_per_node=args.tasks,
        name=args.name,
        slurm_comment=args.comment,
        slurm_constraint=args.constraint,
        nodes=args.nodes,
        tasks_per_node=args.tasks,
        mem_gb=args.mem_gb,
    )

    if args.array:
        executor.update_parameters(slurm_array_parallelism=args.array_parallelism)

    pathlib.Path(args.checkpoint_dir).mkdir(parents=True)

    jobs = []

    if not args.no_preemption:
        combinations = [
            comb + ["--preemptable", f"--checkpoint_freq={args.checkpoint_freq}"]
            for comb in combinations
        ]

    combinations = [
        comb + [f"--checkpoint_dir={args.checkpoint_dir}"] for comb in combinations
    ]

    if args.tasks > 1:
        # ports should be in between 2**10 and 2**16, but we'll start from some random
        # distant offset and use only a part of the space
        # hopefully we'll never have 2**15 jobs on the same node
        combinations = [
            comb + [f"--distributed_port={i % (2**15) + 18363}"]
            for (i, comb) in enumerate(combinations)
        ]

    runner = SlurmWrapper(module.main)
    if not args.array:
        jobs = (executor.submit(runner, comb) for comb in combinations)
    else:
        jobs = executor.map_array(runner, combinations)

    for job, comb in zip(jobs, combinations):
        print(job.job_id, comb)

    print(f"Total jobs launched for total combinations: {len(combinations)}")

    if args.force_requeue:
        time.sleep(60)
        print("sleep over, sending signal")
        for job in jobs:
            print(jobs)
            job._send_requeue_signal(timeout=False)
