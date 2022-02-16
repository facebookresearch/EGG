from pyparsing import cpp_style_comment
from egg.zoo.pop import train
import json
import itertools as it
import os
import sys
import wandb


def write_sequential_jobs(
    json_path,
    logfile_path="/mnt/efs/fs1/logs/",
    prefix="incep_hp_search",
    command_path="~",
):  # doesn't work because of wandb
    with open(command_path, "w") as command_file:
        with open(json_path, "r") as f:
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
                    if allkeys[i] in [
                        "batch_size",
                        "lr",
                        "vocab_size",
                        "recv_hidden_dim",
                    ]:
                        # $batch_size.$lr.$vocab_size.$recv_hidden_dim"
                        expname += f"{arg}."
                outdir = f"{logfile_path}{prefix}.{expname[0:len(expname)-1]}"
                args.append(f"--checkpoint_dir={outdir}")
                # early creation of checkpoint dir for wandb
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                command_file.write(
                    f"sudo python3 egg/zoo/pop/train {' '.join(args)} \\n"
                )


if __name__ == "__main__":
    wandb.config.allow_val_change = True
    args = sys.argv[1:]
    print(args)
    command_file_path = "../bash_exp_plans/hp.sh"
    write_sequential_jobs(
        f"/mnt/efs/fs1/EGG/egg/zoo/pop/paper_sweeps/{args[0]}_hp_search.json",
        prefix=f"{args[0]}_hp_search",
        command_path=command_file_path,
    )
    print(f"experimental plan written to {command_file_path}, launching...")
    os.system(f"sudo bash {command_file_path}")
