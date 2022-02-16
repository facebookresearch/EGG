from egg.zoo.pop import train
import json
import itertools as it
import os
import sys
import wandb


def launch_sequential_jobs(
    json_path, logfile_path="/mnt/efs/fs1/logs/", prefix="incep_hp_search"
):

    with open(json_path) as f:
        params = json.load(f)
        allkeys = sorted(params)
        experiments = it.product(*(params[key] for key in allkeys))
        for experiment in experiments:
            args = []
            expname = ""
            for i, arg in enumerate(experiment):
                if allkeys[i] in ["use_larc", "use_different_architectures"]:
                    args.append(f"--{allkeys[i]}")
                else:
                    args.append(f"--{allkeys[i]}={arg}")
                if allkeys[i] in ["batch_size", "lr", "vocab_size", "recv_hidden_dim"]:
                    # $batch_size.$lr.$vocab_size.$recv_hidden_dim"
                    expname += f"{arg}."
            outdir = f"{logfile_path}{prefix}.{expname[0:len(expname)-2]}"
            args.append(f"--checkpoint_dir={outdir}")
            # early creation of checkpoint dir for wandb
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            train.main(args)


if __name__ == "__main__":
    wandb.config.update(allow_val_change=True)
    args = sys.argv[1:]
    print(args)
    launch_sequential_jobs(
        f"/mnt/efs/fs1/EGG/egg/zoo/pop/paper_sweeps/{args[0]}_hp_search.json",
        prefix=f"{args[0]}_hp_search",
    )
