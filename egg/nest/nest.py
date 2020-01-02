# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import argparse
import time
import importlib
from egg.nest.common import sweep

if __name__ == '__main__':
    from egg.nest.wrappers import SlurmWrapper
    import submitit

    parser = argparse.ArgumentParser(description="nest: a stool-like slurm tool for EGG")
    parser.add_argument("--game", type=str, help="Game's full classpath to run, e.g. egg.zoo.mnist_autoenc.train")
    parser.add_argument("--sweep", action='append', default=[], help="Json file with sweep params in the stool format."
                                                         "It is possible to specify several files: --sweep file1.json --sweep file2.json")
    parser.add_argument("--py_sweep", action='append', default=[], help="A python module, with a grid() method outputting an iterable with the grid parameters."
                        "It is possible to specify several files.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Root folder to save job output")
    parser.add_argument("--preview", action="store_true", help="Don't start jobs, only show the list of commands")
    parser.add_argument("--dry_run", action="store_true", help="Synonym for preview")

    parser.add_argument("--name", type=str, default=None, help="sbatch name of job. Also used in the output directory")
    parser.add_argument("--ncpu", type=int, default=8, help="sbatch number of cpus required per task")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of gpus required per task (--gres=gpu:N in sbatch)")
    parser.add_argument("--partition", type=str, default="dev", help="Partition requested")
    parser.add_argument("--time", type=int, default=4320, help="Job timeout")
    parser.add_argument("--checkpoint_freq", type=int, default=1,
            help="Checkpoint frequency, imposed on an EGG game, in epochs (used to survive preemption). Disabled if set to 0.")
    parser.add_argument("--no_preemption", action="store_true", help="")
    parser.add_argument("--comment", type=str, help="")

    parser.add_argument("--force_requeue", action="store_true", help="Force job requeue after 1 minute [for debug]")

    parser.add_argument("--array", action="store_true", help="Use SLURM arrays")
    parser.add_argument("--array_parallelism", type=int, default=128)


    args = parser.parse_args()

    if not args.game:
        print('--game parameter has to set an EGG-implemented game')
        exit(1)

    if not args.name:
        args.name = args.game.split('.')[-2]

    if args.checkpoint_dir is None:
        args.checkpoint_dir = (pathlib.PosixPath('~/nest') / args.name / time.strftime("%Y_%m_%d_%H_%M_%S")).expanduser()
    module = importlib.import_module(args.game)
    executor = submitit.AutoExecutor(folder=args.checkpoint_dir)
    executor.update_parameters(timeout_min=args.time, partition=args.partition,
            cpus_per_task=args.ncpu, gpus_per_node=args.ngpu, name=args.name, comment=args.comment)

    if args.array:
        executor.update_parameters(array_parallelism=args.array_parallelism)


    combinations = []
    for sweep_file in args.sweep:
        combinations.extend(sweep(sweep_file))

    for py_sweep_file in args.py_sweep:
        sweep_module = importlib.import_module(py_sweep_file)
        combinations.extend(sweep_module.grid())

    pathlib.Path(args.checkpoint_dir).mkdir(parents=True)

    jobs = []

    if not args.no_preemption:
        combinations = [
            comb + ['--preemptable', f'--checkpoint_freq={args.checkpoint_freq}'] for comb in combinations]
    combinations = [comb + [f'--checkpoint_dir={args.checkpoint_dir}'] for comb in combinations]

    if args.preview or args.dry_run:
        for comb in combinations:
            runner = SlurmWrapper(module.main)
            print(f'{comb}')
    elif not args.array:
        for comb in combinations:
            runner = SlurmWrapper(module.main)
            job = executor.submit(runner, comb)
            print(f'job id {job.job_id}, args {comb}')
            jobs.append(job)
    else:
        runner = lambda x: SlurmWrapper(module.main)(x)
        jobs = executor.map_array(runner, combinations)

        for job, comb in zip(jobs, combinations):
            print(job.job_id, comb)

    print(f'Total jobs launched: {len(jobs)}, total combinations: {len(combinations)}')

    if args.force_requeue:
        time.sleep(60)
        print('sleep over, sending signal')
        for job in jobs:
            print(jobs)
            job._send_requeue_signal(timeout=False)

