import os
import json
import itertools as it
import sys
from pathlib import Path
import argparse
import time

default_checkpoint_dir="/homedtcl/mmahaut/projects/experiments"

def get_opts(arguments):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--params_path",
        type=str,
        help="path to the json file containing the parameters to sweep through",
    )
    arg_parser.add_argument(
        "--memory",
        type=str,
        default="32G",
        help="assigned memory in GB",
    )
    arg_parser.add_argument(
        "--job_name",
        type=str,
        default="job",
        help="name of the job. If no checkpoint_dir is given, this is used as the name of the folder in which the job is stored",
    )
    arg_parser.add_argument(
        "--sbatch_dir",
        type=str,
        default="/homedtcl/mmahaut/projects/manual_slurm",
        help="path to the directory where the sbatch file is stored",
    )
    arg_parser.add_argument(
        "--partition",
        type=str,
        default="alien",
        help="slurm partition on which the jobs are run",
    )
    arg_parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of GPUs used per job. Should be 1.",
    )
    arg_parser.add_argument(
        "--time",
        type=str,
        default="3-00:00:00",
        help="time allocated for each job",
    )
    arg_parser.add_argument(
        "--qos",
        type=str,
        default="alien",
        help="slurm qos for each jobs",
    )
    arg_parser.add_argument(
        "--game",
        type=str,
        default="egg.zoo.pop.scripts.analysis_tools.extract_com",
        help="python module path to the game to run",
    )

    opts = arg_parser.parse_args(arguments)
    return opts


def sweep_params(opts):
    with open(opts.params_path, "r") as f:
        params = json.load(f)

        if not "checkpoint_dir" in params :
            params["checkpoint_dir"] = [Path(default_checkpoint_dir)/opts.job_name]
        

        for i, values in enumerate(it.product(*(params[key] for key in params))):
            # diferentiating the checkpoint_dir for each job to avoid overwriting
            params["checkpoint_dir"] = [params["checkpoint_dir"][0] / f"{i}"]
            checkpoint_dir = Path(params["checkpoint_dir"][0])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            command=build_command(opts.game, values, params.keys())
            write_sbatch(command,opts.job_name,opts.sbatch_dir,checkpoint_dir,opts.partition,opts.n_gpus,opts.time,opts.memory,opts.qos)

            sbatch_file = Path(opts.sbatch_dir) / f"{opts.job_name}.sh"
            os.system(f"sbatch {sbatch_file}")
            

def build_command(game, params, keys):
    command=f"python -m {game} "
    for i, key in enumerate(keys):
        if isinstance(params[i], str):
            command += f"--{key}=\"{params[i]}\" "
        elif isinstance(params[i], bool):
            command += f"--{key} " if params[i] else ""
        else:
            command += f"--{key}={params[i]} "
    return command


def write_sbatch(command,jobname,sbatch_dir,checkpoint_dir:Path,partition,n_gpus,time,mem,qos):
    """
    writes a sbatch file for the current job
    """
    sbatch_path = Path(sbatch_dir) / f"{jobname}.sh"
    with open(sbatch_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={checkpoint_dir / jobname}_%j.out
#SBATCH --error={checkpoint_dir / jobname}_%j.err

{command}
echo "done"
"""
        )

if __name__ == "__main__":
    sweep_params(get_opts(sys.argv[1:]))