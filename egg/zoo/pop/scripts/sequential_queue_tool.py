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
    gpu=None,
):  # doesn't work because of wandb
    with open(command_path, "w") as command_file:
        with open(json_path, "r") as f:
            params = json.load(f)
            allkeys = sorted(params)
            cuda = ""
            if gpu is None:
                cuda = ""
            else:
                cuda = f"CUDA_VISIBLE_DEVICES={gpu}"
            experiments = it.product(*(params[key] for key in allkeys))
            for experiment in experiments:
                args = []
                expname = ""
                for i, arg in enumerate(experiment):
                    if allkeys[i] in [
                        "use_different_architectures",
                        "pretrain_vision",
                    ]:
                        args.append(f"--{allkeys[i]}")
                    elif isinstance(arg, str):
                        args.append(f'--{allkeys[i]}="{arg}"')
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
                    f"{cuda} python3 egg/zoo/pop/train.py {' '.join(args)} \n"
                )


if __name__ == "__main__":
    wandb.init()
    wandb.config.allow_val_change = True
    args = sys.argv[1:]
    # [0] is arch type
    # [1] is gpu number
    command_file_path = f"../bash_exp_plans/hp_{args[0]}.sh"
    write_sequential_jobs(
        f"/mnt/efs/fs1/EGG/egg/zoo/pop/paper_sweeps/{args[0]}_hp_search.json",
        prefix=f"{args[0]}_hp_search_p",
        command_path=command_file_path,
        gpu=None if len(args) == 1 else args[1],
    )
    print(f"experimental plan written to {command_file_path}, launching...")
    os.system(f"bash {command_file_path}")
