import os
import json
import itertools as it
import sys

def write_and_run(params, keys, jobname="job",partition="alien",n_gpus=1,time="3-00:00:00",mem="8G"):
    command=f"srun --job-name={jobname} --partition={partition} --gres=gpu:{n_gpus} --nodes=1 --ntasks-per-node=1 --time={time} --mem={mem} python -m egg.zoo.pop.train "
    for i, key in enumerate(keys):
        if isinstance(params[i], str):
            command += f"--{key}=\"{params[i]}\" "
        else:
            command += f"--{key}={params[i]} "
    print(command)
    os.system(command)

def sweep_params(params_path):
    with open(params_path, "r") as f:
        params = json.load(f)
        for _l in it.product(*(params[key] for key in params)):
            write_and_run(_l, [key for key in params])

if __name__ == "__main__":
    sweep_params(params_path=sys.argv[1])