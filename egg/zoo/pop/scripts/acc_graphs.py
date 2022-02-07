import matplotlib.pyplot as plt
import json
import glob
import os


def text_to_data(file_path, mode="train"):  # Mat going through console
    with open(file_path) as f:
        x = []
        y = []
        lines = f.readlines()
        for line in lines:
            if "{" in line:
                _dict = json.loads(line)
                if _dict["mode"] == "train":
                    x.append(_dict["epoch"])
                    y.append(_dict["acc"])
        return x, y


def extract_metadata(path):
    with open(path) as file:
        meta = json.load(file)
        return (
            meta["args"][5][29 : len(meta["args"][5]) - 2]
            + " + "
            + meta["args"][10][31 : len(meta["args"][10]) - 2]
        )


def get_log_files(wandb_path="/mnt/efs/fs1/EGG/wandb"):
    return [file for file in glob.glob(wandb_path + "/run*")]


def make_acc_graph():
    # model identifyer
    # is the validation directly printed in the logs ? if yes could we json bundle it all for easier access ?
    # access the validation data
    # access the number of epochs as x
    # plot with line indicating arrival time at chosen performance
    # Additionaly, give the arriving accuracy and the number of epochs to reach peak performance (to check validity of what will later be used.)
    for i, file_path in enumerate(get_log_files()):
        x, y = text_to_data(os.path.join(file_path, "files/output.log"), mode="train")
        plt.plot(
            x,
            y,
            label=extract_metadata(
                os.path.join(file_path, "files/wandb-metadata.json")
            ),
        )
        plt.legend()
        plt.title("r={}")
        plt.savefig(os.path.join(file_path, "acc_graph.png"))

    pass


make_acc_graph()
