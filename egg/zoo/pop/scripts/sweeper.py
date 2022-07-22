import os
import json
import itertools as it
import sys
from pathlib import Path

def sweep_params(params_path,jobname="job", sbatch_dir="/homedtcl/mmahaut/projects/manual_slurm", partition="alien",n_gpus=1,time="3-00:00:00",mem="8G"):
    with open(params_path, "r") as f:
        params = json.load(f)
        for _l in it.product(*(params[key] for key in params)):
            command=build_command(_l, params.keys())
            checkpoint_dir = Path(params["checkpoint_dir"]) if "checkpoint_dir" in params else Path(sbatch_dir)
            write_sbatch(command,jobname,sbatch_dir,checkpoint_dir,partition,n_gpus,time,mem)
            sbatch_file = Path(sbatch_dir) / f"{jobname}.sh"
            os.system(f"sbatch {sbatch_file}")
            

def build_command(params, keys):
    command=f"python -m egg.zoo.pop.train "
    for i, key in enumerate(keys):
        if isinstance(params[i], str):
            command += f"--{key}=\"{params[i]}\" "
        elif isinstance(params[i], bool):
            command += f"--{key} " if params[i] else ""
        else:
            command += f"--{key}={params[i]} "
    return command


def write_sbatch(command,jobname,sbatch_dir,checkpoint_dir:Path,partition="alien",n_gpus=1,time="3-00:00:00",mem="8G"):
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
        f.write("\n".join(sys.argv))

if __name__ == "__main__":
    # this is not as clean as something like itertools, but it works for now
    sweep_params(params_path=sys.argv[1], jobname=sys.argv[2] if len(sys.argv)>2 else "job", sbatch_dir=sys.argv[3] if len(sys.argv)>3 else "/homedtcl/mmahaut/projects/manual_slurm", partition=sys.argv[4] if len(sys.argv)>4 else "alien",n_gpus=sys.argv[5] if len(sys.argv)>5 else 1,time=sys.argv[6] if len(sys.argv)>6 else "3-00:00:00",mem=sys.argv[7] if len(sys.argv)>7 else "8G")