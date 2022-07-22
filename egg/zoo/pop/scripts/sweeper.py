import os
import json
import itertools as it
import sys

def write_and_run(params, keys, jobname="job",partition="alien",n_gpus=1,time="3-00:00:00",mem="8G"):
    command=f"srun --job-name={jobname} --partition={partition} --gres=gpu:{n_gpus} --nodes=1 --ntasks-per-node=1 --time={time} --mem={mem} python -m egg.zoo.pop.train "
    for i, key in enumerate(keys):
        if isinstance(params[i], str):
            command += f"--{key}=\"{params[i]}\" "
        elif isinstance(params[i], bool):
            command += f"--{key} " if params[i] else ""
        else:
            command += f"--{key}={params[i]} "
    print(command)
    os.system(command)

def sweep_params(params_path,jobname="job",partition="alien",n_gpus=1,time="3-00:00:00",mem="8G"):
    with open(params_path, "r") as f:
        params = json.load(f)
        for _l in it.product(*(params[key] for key in params)):
            write_and_run(_l, [key for key in params],jobname,partition,n_gpus,time,mem)

if __name__ == "__main__":
    # this is not as clean as something like itertools, but it works for now
    sweep_params(params_path=sys.argv[1], jobname=sys.argv[2] if len(sys.argv)>2 else "job",partition=sys.argv[3] if len(sys.argv)>3 else "alien",n_gpus=sys.argv[4] if len(sys.argv)>4 else 1,time=sys.argv[5] if len(sys.argv)>5 else "3-00:00:00",mem=sys.argv[6] if len(sys.argv)>6 else "8G")