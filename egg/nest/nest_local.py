# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import argparse
import time
import importlib
from concurrent.futures import ProcessPoolExecutor, wait

from egg.nest.common import sweep
from egg.nest.wrappers import ConcurrentWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nest_local: a simple grid-search tool for EGG")
    parser.add_argument("--game", type=str, help="Game's full classpath to run, e.g. egg.zoo.mnist_autoenc.train")
    parser.add_argument("--sweep", action='append', help="Json file with sweep params in the stool format."
                        "It is possible to specify several files: --sweep file1.json --sweep file2.json")
    parser.add_argument("--root_dir", type=str, default=None, help="Root folder to save the output")
    parser.add_argument("--name", type=str, default=None, help="Name for the run")

    parser.add_argument("--preview", action="store_true", help="Don't start jobs, only show the list of commands")
    parser.add_argument("--dry_run", action="store_true", help="Synonym for preview")

    parser.add_argument("--n_workers", type=int, default=1, help="Number of active worker jobs")
    parser.add_argument("--checkpoint_freq", type=int, default=0, help="Checkpoint frequency, in epochs")

    args = parser.parse_args()

    if not args.game:
        print('--game parameter has to set an EGG-implemented game')
        exit(1)

    if not args.name:
        args.name = args.game.split('.')[-2]

    if args.root_dir is None:
        args.root_dir = (pathlib.PosixPath('~/nest_local') / args.name / time.strftime("%Y_%m_%d_%H_%M_%S")).expanduser()

    module = importlib.import_module(args.game)
    pathlib.Path(args.root_dir).mkdir(parents=True)

    jobs = []

    comb_id = -1
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for sweep_file in args.sweep:
            for comb_id, comb in enumerate(sweep(sweep_file), start=comb_id + 1):
                runner = ConcurrentWrapper(runnable=module.main,
                                           log_dir=args.root_dir,
                                           job_id=comb_id)

                if args.checkpoint_freq > 0:
                    checkpoint_dir = pathlib.Path(args.root_dir) / f"{comb_id}"
                    checkpoint_dir.mkdir()

                    comb.extend([f"--checkpoint_freq={args.checkpoint_freq}",
                                 f"--checkpoint_dir={checkpoint_dir}"])

                if not args.preview and not args.dry_run:
                    job = executor.submit(runner, comb)
                    jobs.append(job)
                print(f'{" ".join(comb)} -> {args.root_dir}/{comb_id}.{{out,err}}')

    print(f'Jobs launched: {len(jobs)}')
    wait(jobs)
