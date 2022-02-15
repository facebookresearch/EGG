from egg.zoo.pop import train
import json
import itertools as it
import os


def launch_sequential_jobs(
    json_path, logfile_path="/mnt/efs/fs1/logs/", prefix="incep_hp_search"
):

    with open(json_path) as f:
        params = json.load(f)
        allkeys = sorted(params)
        experiments = it.product(*(params[key] for key in allkeys))
        for experiment in experiments:
            args = []
            for i, arg in enumerate(experiment):
                args.append(f"--{allkeys[i]}={arg}")
            outdir = f"{logfile_path}{prefix}.{outdir[0:len(outdir-2)]}"
            args.append(f"--checkpoint_dir={outdir}")
            # early creation of checkpoint dir for wandb
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            train.main(args)


if __name__ == "__main__":
    launch_sequential_jobs(
        "/mnt/efs/fs1/EGG/egg/zoo/pop/paper_sweeps/incep_hp_search.json"
    )
