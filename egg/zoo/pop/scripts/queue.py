from egg.zoo.pop.train import main
import json
import itertools as it


def launch_sequential_jobs(json_path):
    with open(json_path) as f:
        params = json.load(f)
        allkeys = sorted(params)
        experiments = it.product(*(params[key] for key in allkeys))
        for experiment in experiments:
            args = []
            for i, arg in enumerate(experiment):
                args.append(f"--{allkeys[i]}={arg}")
            main(args)


if __name__ == "__main__":
    launch_sequential_jobs(
        "/mnt/efs/fs1/EGG/egg/zoo/pop/paper_sweeps/incep_hp_search.json"
    )
